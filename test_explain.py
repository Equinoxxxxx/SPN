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
from tools.data.normalize import recover_norm_imgs, img_mean_std_BGR, recover_norm_sklt, recover_norm_bbox

from get_args import get_args

torch.backends.mha.set_fastpath_enabled(False)

import matplotlib.pyplot as plt
import numpy as np
from tools.data.resize_img import resize_image

def plot_all_explanation(explain_info, path, part_height=200, part_width=200, spacing=50):
    '''
    explain_info: P*[{'mean_rel_var':float, 
                      'last_weights':{act:array}, 
                      'sample_info':[{'rel_var':float,
                                      'labels':{act:int}, 
                                      'images':{modality:array}, 
                                      'modality_effs':{modality:float}, 
                                      'proto_simi':float,
                                      }]
                      }]
    path: str, the path to save the plot
    row_spacing: float, the spacing between rows
    col_spacing: float, the spacing between columns
    '''

    P = len(explain_info)
    K = len(explain_info[0]['sample_info'])
    M = len(explain_info[0]['sample_info'][0]['images'])

    # Calculate the total height and width of the canvas
    total_height = P * K * (part_height + spacing)
    total_width = (M+3) * (part_width + spacing)

    # Create a blank canvas
    canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

    for i, block in enumerate(explain_info):
        block_y_offset = i * K * (part_height + spacing)
        
        # Draw mean_rel_var
        mean_rel_var = block['mean_rel_var']
        proto_title = f'prototype {i}'
        mean_rel_var_text = f"mean_rel_var: {mean_rel_var}"
        cv2.putText(canvas, proto_title, (spacing, block_y_offset + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, mean_rel_var_text, (spacing, block_y_offset + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Draw sample info
        for j, sample in enumerate(block['sample_info']):
            sample_y_offset = block_y_offset + j * (part_height + spacing)
            sample_left_text = [f'sample {j}', 
                                f"rel_var: {sample['rel_var']}", 
                                'labels:'
                                ]
            for act in sample['labels']:
                sample_left_text.append(f"{act}: {sample['labels'][act]}")
            for l in range(len(sample_left_text)):
                cv2.putText(canvas, sample_left_text[l], (spacing+15, sample_y_offset + part_height//4 + l*15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            for k, (modality, image) in enumerate(sample['images'].items()):
                img_x_offset = spacing + (k + 1) * (part_width + spacing)
                img_y_offset = sample_y_offset
                img_resized = resize_image(image, (part_width, part_height), mode='pad', padding_color=(255, 255, 255))
                try:
                    canvas[img_y_offset:img_y_offset + part_height, img_x_offset:img_x_offset + part_width] = img_resized
                except:
                    import pdb; pdb.set_trace()
                modality_eff_text = f"modality eff: {sample['modality_effs'][modality]}"
                cv2.putText(canvas, modality, (img_x_offset, img_y_offset + part_height + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(canvas, modality_eff_text, (img_x_offset, img_y_offset + part_height + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            proto_simi_text = f"proto_simi: {sample['proto_simi']}"
            cv2.putText(canvas, proto_simi_text, (spacing + (M + 1) * (part_width + spacing), sample_y_offset + part_height // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Draw last_weights
        weights_y_offset = block_y_offset + K * (part_height + spacing) // 2
        for act, cls in block['last_weights'].items():
            weights_text = f"{act}: {cls}"
            cv2.putText(canvas, weights_text, (spacing + (M + 2) * (part_width + spacing), weights_y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            weights_y_offset += spacing
    
    # Save the final image
    cv2.imwrite(path, canvas)

    return canvas

# example
explain_info = [
    {
        'mean_rel_var': 0.1,
        'last_weights': {'act1': np.array([0.2, 0.3]), 'act2': np.array([0.4, 0.5])},
        'sample_info': [
            {
                'rel_var': 0.05,
                'labels': {'act1': 0, 'act2': 1},
                'images': {'mod1': np.random.rand(64, 64, 3), 'mod2': np.random.rand(64, 64, 3)},
                'modality_effs': {'mod1': 0.8, 'mod2': 0.6},
                'proto_simi': 0.9
            },
            {
                'rel_var': 0.07,
                'labels': {'act1': 0, 'act2': 1},
                'images': {'mod1': np.random.rand(64, 64, 3), 'mod2': np.random.rand(64, 64, 3)},
                'modality_effs': {'mod1': 0.7, 'mod2': 0.5},
                'proto_simi': 0.85
            }
        ]
    },
    {
        'mean_rel_var': 0.1,
        'last_weights': {'act1': np.array([0.2, 0.3]), 'act2': np.array([0.4, 0.5])},
        'sample_info': [
            {
                'rel_var': 0.05,
                'labels': {'act1': 0, 'act2': 1},
                'images': {'mod1': np.random.rand(64, 64, 3), 'mod2': np.random.rand(64, 64, 3)},
                'modality_effs': {'mod1': 0.8, 'mod2': 0.6},
                'proto_simi': 0.9
            },
            {
                'rel_var': 0.07,
                'labels': {'act1': 0, 'act2': 1},
                'images': {'mod1': np.random.rand(64, 64, 3), 'mod2': np.random.rand(64, 64, 3)},
                'modality_effs': {'mod1': 0.7, 'mod2': 0.5},
                'proto_simi': 0.85
            }
        ]
    }
]

plot_all_explanation(explain_info, 'explanation.png')