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
from get_args import get_args, process_args

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

def kmph2mpss(kmph):
    pass
    

if __name__ == '__main__':
    seed_all(42)
    # parser = argparse.ArgumentParser(description='main')
    # parser.add_argument('--a', type=str, default='a')
    # parser.add_argument('--b', type=int, default=0)

    # args = parser.parse_args()
    
    # args = get_args()
    # dataset = TITAN_dataset(sub_set='default_train', 
    #                             offset_traj=False,
    #                             img_norm_mode=args.img_norm_mode, 
    #                             target_color_order=args.model_color_order,
    #                             obs_len=args.obs_len, 
    #                             pred_len=args.pred_len, 
    #                             overlap_ratio=0.5, 
    #                             obs_fps=args.obs_fps,
    #                             recog_act=False,
    #                             multi_label_cross=False, 
    #                             act_sets=args.act_sets,
    #                             loss_weight='sklearn',
    #                             small_set=0,
    #                             resize_mode=args.resize_mode, 
    #                             modalities=args.modalities,
    #                             img_format=args.img_format,
    #                             sklt_format=args.sklt_format,
    #                             ctx_format=args.ctx_format,
    #                             traj_format=args.traj_format,
    #                             ego_format=args.ego_format,
    #                             augment_mode=args.augment_mode,
    #                             )
    # Sample args1 and args2 creation

    from tools.visualize.visualize_1d_seq import vis_2_1d_seqs
    proto0_vel = [
        [20.00414592,20.00414592,20.00414592,20.00414592,20.00414592,
         20.00414592,20.00414592,19.006352640000003,18.00855936,18.00855936,
         18.00855936,18.00855936,18.00855936,15.99687936,15.99687936,
         15.99687936,15.99687936,15.99687936,15.99687936,15.99687936,
         15.99687936,15.99687936,15.99687936,15.99687936,15.99687936,14.0012928,
         14.0012928,14.0012928,12.00570624,12.00570624,12.00570624,
         12.00570624,12.00570624,12.00570624,12.00570624,12.00570624,
         12.00570624,10.01011968,10.01011968,10.01011968,10.01011968,
         10.01011968,10.01011968,10.01011968,10.01011968,10.01011968,6.00285312,
         6.00285312,6.00285312,6.00285312,6.00285312,6.00285312,
         6.00285312,6.00285312,6.00285312,6.00285312,6.00285312,
         6.00285312,5.00505984,4.007266560000001], # 4 3 5769-5828
        [], # 4 1 3793-3852
        [], # 4 1 3793-3852
        [], # 4 12 6564-6623
        [], # 4 12 6564-6623
    ]
    proto5_vel = [
        [], # 4 10 13681-13740
        [], # 4 4 414-473
        [], # 2 2 14034-14093
        [], # 1 2 4573-4632
        [], # 4 7 6725-6784
    ]

    proto0_accel = [
    ]
    
    # Example with 4:1 ratio
    long_seq = np.random.randn(10)  # Length 100
    short_seq = np.random.randn(5)  # Length 25
    seqs = [long_seq, short_seq, ]
    tags = ['High-res signal', 'Down-sampled signal']
    lims = [(-3, 3), (-2, 2)]
    img = vis_2_1d_seqs(seqs, tags, lims, 'test_output/output.png', weights=None, mode='below')
    print(np.linspace(0, 10, 5))