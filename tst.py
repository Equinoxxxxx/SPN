from config import dataset_root
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pdb
import os
import argparse
import os
import cv2
import sys
py_dll_path = os.path.join(sys.exec_prefix, 'Library', 'bin')
os.environ['PATH'] += py_dll_path
from models.backbones import create_backbone, CustomTransformerEncoder1D
from models.mytransformer import MyTransformerEncoder,MyTransformerEncoderLayer,\
    set_attn_args,SaveOutput
from tools.visualize.visualize_seg_map import visualize_segmentation_map
from torchvision import transforms
from PIL import Image
from tools.data.normalize import norm_imgs, recover_norm_imgs, img_mean_std_BGR
from tools.utils import save_model, seed_all

torch.backends.mha.set_fastpath_enabled(False)

def A():
    raise IndexError()

class D(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.samples = torch.range(0, 10)
    
    def __getitem__(self, idx):
        # A()
        raise IndexError()
        return self.samples[idx]
    def __len__(self):
        return len(self.samples)
    

class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class MM(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=4,
                                                                    nhead=4,
                                                                    dim_feedforward=3,
                                                                    activation="gelu",
                                                                    batch_first=True), 
                                        num_layers=1)
        # register hook
        for n,m in self.transformer.named_modules():
            if isinstance(m, nn.MultiheadAttention):
                set_attn_args(m)
                m.register_forward_hook(self.get_attn)
        self.attn_list = []
    def forward(self, x):
        self.attn_list = []
        x = self.transformer(x)
        print('attn num:', len(self.attn_list))
        try:
            print('attn shape:', self.attn_list[0].size())
        except:
            import pdb;pdb.set_trace()
        return x
    def get_attn(self, module, input, output):
        self.attn_list.append(output[1].clone().detach().cpu())

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.samples = torch.rand(10, 4)
    def __getitem__(self, idx):
        return self.samples[idx]
    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':
    seed_all(42)
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--a', type=str, default='a')
    parser.add_argument('--b', type=int, default=0)

    args = parser.parse_args()
    
    # cur_crop_seg_root = '/home/y_feng/workspace6/datasets/BDD100k/bdd100k/extra/cropped_seg/person/ori_local/224w_by_224h/ped'
    # cur_seg_oids = set(os.listdir(cur_crop_seg_root))
    # cur_crop_img_root = '/home/y_feng/workspace6/datasets/BDD100k/bdd100k/extra/cropped_images/even_padded/224w_by_224h/ped'
    # cur_crop_img_oids = set(os.listdir(cur_crop_img_root))
    # cur_crop_ctx_root = '/home/y_feng/workspace6/datasets/BDD100k/bdd100k/extra/context/ori_local/224w_by_224h/ped'
    # cur_crop_ctx_oids = set(os.listdir(cur_crop_ctx_root))
    # print(f'cur_crop_seg - cur_crop_img: {cur_seg_oids - cur_crop_img_oids}')
    # print(f'cur_crop_img - cur_crop_seg: {cur_crop_img_oids - cur_seg_oids}')
    # print(f'cur_crop_ctx - cur_crop_img: {cur_crop_ctx_oids - cur_crop_img_oids}')
    # print(f'cur_crop_img - cur_crop_ctx: {cur_crop_img_oids - cur_crop_ctx_oids}')
    # oids_to_add = cur_crop_img_oids - cur_seg_oids
    # oids_to_add.add('112811')
    # with open('./oids_to_add_bdd100k.pkl', 'wb') as f:
    #     pickle.dump(oids_to_add, f)
    a = 0
    a += np.array([1,2,3])
    print(a)