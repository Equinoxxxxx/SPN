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
from tools.data.normalize import norm_imgs, recover_norm_imgs, img_mean_std_BGR, sklt_local_to_global\
    , sklt_global_to_local
from tools.utils import save_model, seed_all
from tools.visualize.visualize_1d_seq import vis_1d_seq, generate_colormap_legend

from tools.datasets.TITAN import TITAN_dataset
from tools.datasets.identify_sample import get_ori_img_path
from tools.visualize.visualize_neighbor_bbox import visualize_neighbor_bbox
from tools.visualize.visualize_skeleton import visualize_sklt_with_pseudo_heatmap
from tools.data.normalize import recover_norm_imgs, img_mean_std_BGR, recover_norm_sklt, recover_norm_bbox
from tools.data.resize_img import resize_image
from get_args import get_args

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
    # parser = argparse.ArgumentParser(description='main')
    # parser.add_argument('--a', type=str, default='a')
    # parser.add_argument('--b', type=int, default=0)

    # args = parser.parse_args()
    
    # args = get_args()
    # dataset = TITAN_dataset(sub_set='default_train', 
    #                                         offset_traj=False,
    #                                         img_norm_mode=args.img_norm_mode, 
    #                                         target_color_order=args.model_color_order,
    #                                         obs_len=args.obs_len, 
    #                                         pred_len=args.pred_len, 
    #                                         overlap_ratio=0.5, 
    #                                         obs_fps=args.obs_fps,
    #                                         recog_act=False,
    #                                         multi_label_cross=False, 
    #                                         act_sets=args.act_sets,
    #                                         loss_weight='sklearn',
    #                                         small_set=0.01,
    #                                         resize_mode=args.resize_mode, 
    #                                         modalities=args.modalities,
    #                                         img_format=args.img_format,
    #                                         sklt_format=args.sklt_format,
    #                                         ctx_format=args.ctx_format,
    #                                         traj_format=args.traj_format,
    #                                         ego_format=args.ego_format,
    #                                         augment_mode=args.augment_mode,
    #                                         )
    # d = dataset[0]
    # set_id = d['set_id_int'].detach().numpy()
    # vid_id = d['vid_id_int'].detach().numpy()
    # img_nms = d['img_nm_int'].detach().numpy()
    # obj_id = d['ped_id_int'].detach().numpy()
    # dataset_name = 'TITAN'
    # traj = d['obs_bboxes_unnormed'].detach().numpy()
    # print(f'traj {traj}')
    # if '0-1' in args.traj_format:
    #     traj = recover_norm_bbox(traj, dataset_name)  # T 4 (int)
    # traj = traj.astype(np.int32)
    # # print(traj)
    # neighbor_bbox = d['obs_neighbor_bbox'].int().detach().numpy() # K T 4
    # K = neighbor_bbox.shape[0]
    # weights = np.arange(K)
    # bg_img_path = get_ori_img_path(dataset,
    #                                 set_id=set_id,
    #                                 vid_id=vid_id,
    #                                 img_nm=img_nms[-1],
    #                                 )
    # bg_img = cv2.imread(bg_img_path)
    # social_img = visualize_neighbor_bbox(bg_img,traj[-1],neighbor_bbox[:,-1], weights=weights)

    # cv2.imwrite('social_img.jpg', social_img)




    T = 5  # Number of frames
    H, W = 384, 288  # Image dimensions
    nj = 17  # Number of joints

    # Generate random images
    imgs = np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8)

    # Generate random skeletons
    sklts = torch.randint(0, min(H, W), (T, nj, 2))
    sklts[0,:,0] = 80
    sklts[0,:,1] = torch.arange(nj) / nj * H
    sklts = sklts.int()
    # print(f'sklts {sklts}')

    # Generate random features
    feats = torch.rand(T, nj).numpy() * nj
    feats[0,:] = torch.arange(nj)
    # print(f'feats {feats}')
    # Generate random bounding boxes
    bboxs = []
    for t in range(T):
        bbox = torch.tensor([0, 0, W, H]) * (t+1) / T
        bbox = bbox.int()
        bboxs.insert(0, bbox)
    bboxs = torch.stack(bboxs).numpy()
    # Define dataset name and save directory
    dataset_name = 'PIE'
    save_dir = 'test_output'
    os.makedirs(save_dir, exist_ok=True)

    # Call the function
    overlay_imgs, heatmaps = visualize_sklt_with_pseudo_heatmap(imgs, sklts, feats, bboxs, dataset_name, save_dir)
    import pdb;pdb.set_trace()
    # print(heatmaps)

    # a = torch.arange(24).reshape(2,3,4)
    # print(a)
    # a[:,:,0] = 1
    # print(a)
    # a[0,:,1] = torch.arange(3)
    # print(a)
