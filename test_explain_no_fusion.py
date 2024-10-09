import torch
import numpy as np
from tqdm import tqdm
import os
import cv2
import copy

from tools.utils import makedir, write_info_txt
from tools.datasets.identify_sample import get_ori_img_path, get_sklt_img_path,\
    DATASET_TO_ID, ID_TO_DATASET, MODALITY_TO_ID, ID_TO_MODALITY, \
        LABEL_TO_CROSSING, LABEL_TO_ATOMIC_CHOSEN, LABEL_TO_SIMPLE_CONTEXTUAL, LABEL_TO_COMPLEX_CONTEXTUAL, \
            LABEL_TO_COMMUNICATIVE, LABEL_TO_TRANSPORTIVE, LABEL_TO_AGE
from tools.data.normalize import recover_norm_imgs, img_mean_std_BGR, recover_norm_sklt, recover_norm_bbox
from tools.visualize.heatmap import visualize_featmap3d
from tools.visualize.visualize_skeleton import visualize_sklt_with_pseudo_heatmap
from tools.visualize.visualize_bbox import draw_boxes_on_img
from tools.visualize.visualize_1d_seq import vis_1d_seq
from tools.visualize.visualize_neighbor_bbox import visualize_neighbor_bbox
from tools.data.resize_img import resize_image

from explain_no_fusion import select_topk_no_fusion, forwad_pass_no_fusion
from get_args import get_args, process_args
from main import construct_data_loader, construct_model


def main():
    log = print
    args = process_args(get_args())
    
    args.dataset_names = [['TITAN'], ['TITAN']]
    args.test_dataset_names = [['TITAN'], ['TITAN']]
    args.small_set = 0.02
    args.ctx_backbone_name = 'pedgraphconv'
    args.ctx_format = 'ped_graph'
    save_root = '../exp_dir/test'
    makedir(save_root)
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load the data
    print('----------------------------Construct data loaders-----------------------------')
    train_loaders, val_loaders, test_loaders = construct_data_loader(args)
    # construct the model
    log('----------------------------Construct model-----------------------------')
    model = construct_model(args, device)
    model = model.float().to(device)
    model_parallel = torch.nn.parallel.DataParallel(model)
    model_parallel.eval()
    select_topk_no_fusion(dataloader=train_loaders[0], 
                            model_parallel=model_parallel, 
                            args=args, 
                            device=device,
                            modalities=args.modalities,
                            save_root=save_root,
                            log=log)


if __name__ == '__main__':
    main()