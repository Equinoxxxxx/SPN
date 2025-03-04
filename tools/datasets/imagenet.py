import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from enum import Enum
import random

class ImageNetTask(Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    LOCALIZATION = "localization"

class ImageNetDataset(Dataset):
    def __init__(
        self,
        root_dir,
        task=ImageNetTask.CLASSIFICATION,
        transform=None,
        full_dataset=False,
        subset_fraction=1.0,
        split="train"
    ):
        """
        Args:
            root_dir (str): Directory with all the images and annotations
            task (ImageNetTask): Which task to load data for
            transform: Optional transform to be applied on images
            full_dataset (bool): Whether to load full ImageNet or ILSVRC subset
            subset_fraction (float): Fraction of dataset to load (0.0-1.0)
            split (str): 'train', 'val', or 'test'
        """
        self.root_dir = root_dir
        self.task = task
        self.transform = transform or transforms.ToTensor()
        self.full_dataset = full_dataset
        self.split = split
        
        # Load appropriate annotation file based on task and split
        self.annotations = self._load_annotations()
        
        # Subsample if needed
        if subset_fraction < 1.0:
            n_samples = int(len(self.annotations) * subset_fraction)
            self.annotations = random.sample(self.annotations, n_samples)
            
        # Load class mapping
        self.class_to_idx = self._load_class_mapping()
        
    def _load_annotations(self):
        """Load annotations based on task and split"""
        anno_path = os.path.join(
            self.root_dir,
            "annotations",
            f"{self.split}_{self.task.value}.json"
        )
        with open(anno_path, 'r') as f:
            return json.load(f)
            
    def _load_class_mapping(self):
        """Load class name to index mapping"""
        mapping_file = os.path.join(self.root_dir, "class_mapping.json")
        with open(mapping_file, 'r') as f:
            return json.load(f)
            
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, idx):
        anno = self.annotations[idx]
        img_path = os.path.join(self.root_dir, "images", anno["image_path"])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Prepare output based on task
        if self.task == ImageNetTask.CLASSIFICATION:
            label = self.class_to_idx[anno["class"]]
            return image, label
            
        elif self.task == ImageNetTask.DETECTION:
            boxes = torch.tensor(anno["boxes"])
            labels = torch.tensor([self.class_to_idx[c] for c in anno["classes"]])
            return image, {"boxes": boxes, "labels": labels}
            
        elif self.task == ImageNetTask.LOCALIZATION:
            box = torch.tensor(anno["box"])
            label = self.class_to_idx[anno["class"]]
            return image, {"box": box, "label": label}