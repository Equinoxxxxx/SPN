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
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--a', type=str, default='a')
    parser.add_argument('--b', type=int, default=0)

    args = parser.parse_args()
    
    args = get_args()
    dataset = TITAN_dataset(sub_set='default_train', 
                                offset_traj=False,
                                img_norm_mode=args.img_norm_mode, 
                                target_color_order=args.model_color_order,
                                obs_len=args.obs_len, 
                                pred_len=args.pred_len, 
                                overlap_ratio=0.5, 
                                obs_fps=args.obs_fps,
                                recog_act=False,
                                multi_label_cross=False, 
                                act_sets=args.act_sets,
                                loss_weight='sklearn',
                                small_set=0,
                                resize_mode=args.resize_mode, 
                                modalities=args.modalities,
                                img_format=args.img_format,
                                sklt_format=args.sklt_format,
                                ctx_format=args.ctx_format,
                                traj_format=args.traj_format,
                                ego_format=args.ego_format,
                                augment_mode=args.augment_mode,
                                )
    n_0_w = 0
    r_1520 = 0
    for d in dataset:
        traj_ori = d['obs_bboxes']
        vid_id = d['vid_id_int']
        obj_id = d['ped_id_int']
        for i in range(len(traj_ori)):
            bb = d['obs_bboxes'][i]
            bb_unnormed = d['obs_bboxes_unnormed'][i]
            bb_ori = d['obs_bboxes_ori'][i]
            img_nm = d['img_nm_int'][i]
            l, t, r, b = bb_ori
            w = r-l
            h = b-t
            # if r > 1520:
            #     r_1520 += 1
            #     print('r>1520:', bb)
            #     print(f'{vid_id}_{obj_id}_{img_nm}')
            if w <= 0:
                n_0_w += 1
                print('w:', bb)
                print(f'{vid_id}_{obj_id}_{img_nm}')
                print(f'bb: {bb} bb unnormed: {bb_unnormed} bb ori: {bb_ori}')

    print(f'len(dataset): {len(dataset)}')  # 6089
    print(f'n_0_w: {n_0_w}')  # 1273
    print(f'r_1520: {r_1520}')  # 0