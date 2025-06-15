import os
import pickle
import shutil
import time
from turtle import resizemode

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
import re
import copy
import numpy as np
from tqdm import tqdm

from tools.datasets.PIE_JAAD import PIEDataset
from tools.datasets.TITAN import TITAN_dataset
from tools.utils import makedir, draw_curves
from tools.log import create_logger
from tools.metrics import *
from tools.visualize.plot import draw_morf

from tools.datasets.construct_dataloader import construct_data_loader
from models.construct_model import construct_model

def test():
    parser = argparse.ArgumentParser()
    print('start')
    # general setting
    parser.add_argument('--ckpt_path', 
                        type=str, 
                        default=\
                        '/home/y_feng/workspace6/work/PedSpace/exp_dir/SLENN_img_sklt_ctx_traj_ego/exp764/ckpt/4_0.0000.pth'
                            )
    parser.add_argument('--args_path', 
                        type=str, 
                        default=\
                        '/home/y_feng/workspace6/work/PedSpace/exp_dir/SLENN_img_sklt_ctx_traj_ego/exp764/args.pkl'
                            )

    test_args = parser.parse_args()
    ckpt_path = test_args.ckpt_path
    args_path = test_args.args_path

    with open(args_path, 'rb') as f:
        args = pickle.load(f)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Default device: ', os.environ['CUDA_VISIBLE_DEVICES'])

    # dirs
    model_dir = os.path.join('/', *ckpt_path.split('/')[:-2])
    test_id = time.strftime("%d%b%Y-%Hh%Mm%Ss")
    test_dir = os.path.join(model_dir, 'test', test_id)
    makedir(test_dir)

    # logger
    log, logclose = create_logger(log_filename=os.path.join(test_dir, 'test.log'))
    log('--------test_args----------')
    for k in list(vars(test_args).keys()):
        log(str(k)+': '+str(vars(test_args)[k]))
    log('--------test_args----------\n')
    log('--------args----------')
    for k in list(vars(args).keys()):
        log(str(k)+': '+str(vars(args)[k]))
    log('--------args----------\n')

    # load the data
    log('----------------------------Load data-----------------------------')
    # test set
    train_loaders, val_loaders, test_loaders = construct_data_loader(args, log)

    # construct the model
    log('----------------------------Construct model-----------------------------')
    model = construct_model(args, device)
    model = model.float().to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model_parallel = torch.nn.DataParallel(model)
    model_parallel.eval()

    # train set
    # n_sample, n_concept
    all_simis = forward_pass(model_parallel, 
                             train_loaders[0], 
                             model_name=args.model_name, 
                             device=device, 
                             modalities=args.modalities, 
                             mask=None, 
                             log=log)
    # select topk samples
    simi_mean = all_simis.mean(dim=0)  # (P)
    simi_var = all_simis.var(dim=0, unbiased=True)  # (P)
    all_relative_var = (all_simis - simi_mean)**2 \
        / (simi_var.unsqueeze(0) + 1e-5)  # (n_samples, P)
    for topk_metric in ['relative_var', 'activation']:
        # top_k_relative_var, top_k_rel_var_indices = torch.topk(all_relative_var, args.topk_explain, dim=0)  # (k, P)
        if topk_metric == 'relative_var':
            top_k_relative_var, top_k_indices = torch.topk(all_relative_var, args.topk_explain, dim=0)  # (k, P)
        elif topk_metric == 'activation':
            _, top_k_indices = torch.topk(all_simis, args.topk_explain, dim=0)  # (k, P)
            top_k_relative_var = torch.gather(all_relative_var, 0, top_k_indices)  # (k, P)
        else:
            raise ValueError(topk_metric)
        log(f'{args.model_name}  {args.dataset_names[0]} train {topk_metric}  {top_k_relative_var.mean()}')
    # test set
    for test_loader in test_loaders[0]:
        cur_dataset = test_loader.dataset.dataset_name
        log(cur_dataset)
        all_simis = forward_pass(model_parallel, 
                                 test_loader, 
                                 model_name=args.model_name, 
                                 device=device, 
                                 modalities=args.modalities, 
                                 mask=None, 
                                 log=log)
        # select topk samples
        simi_mean = all_simis.mean(dim=0)  # (P)
        simi_var = all_simis.var(dim=0, unbiased=True)  # (P)
        all_relative_var = (all_simis - simi_mean)**2 \
            / (simi_var.unsqueeze(0) + 1e-5)  # (n_samples, P)
        for topk_metric in ['relative_var', 'activation']:  
            if topk_metric == 'relative_var':
                top_k_relative_var, top_k_indices = torch.topk(all_relative_var, args.topk_explain, dim=0)  # (k, P)
            elif topk_metric == 'activation':
                _, top_k_indices = torch.topk(all_simis, args.topk_explain, dim=0)  # (k, P)
                top_k_relative_var = torch.gather(all_relative_var, 0, top_k_indices)  # (k, P)
            else:
                raise ValueError(topk_metric)
            log(f'{args.model_name}  {cur_dataset} test {topk_metric}  {top_k_relative_var.mean()}')


def forward_pass(model, 
                 dataloader, 
                 model_name, 
                 device, 
                modalities=['img'],
                mask=None,
                log=print,
                ):
    start = time.time()
    # targets and logits for whole epoch
    targets_e = {}
    logits_e = {}
    relevs_e = []

    # start iteration
    b_end = time.time()
    tbar = tqdm(dataloader, miniters=1)
    all_simis = []
    for iter, data in enumerate(tbar):
        # load inputs
        inputs = {}
        if 'img' in modalities:
            inputs['img'] = data['ped_imgs'].to(device)
        if 'skeleton' in modalities:
            inputs['skeleton'] = data['obs_skeletons'].to(device)
        if 'context' in modalities:
            inputs['context'] = data['obs_context'].to(device)
        if 'traj' in modalities:
            inputs['traj'] = data['obs_bboxes'].to(device)
        if 'ego' in modalities:
            inputs['ego'] = data['obs_ego'].to(device)

        # load gt
        targets = {}
        targets['cross'] = data['pred_act'].to(device).view(-1) # idx, not one hot
        targets['atomic'] = data['atomic_actions'].to(device).view(-1)
        targets['complex'] = data['complex_context'].to(device).view(-1)
        targets['communicative'] = data['communicative'].to(device).view(-1)
        targets['transporting'] = data['transporting'].to(device).view(-1)
        targets['age'] = data['age'].to(device).view(-1)

        # forward
        b_start = time.time()
        # torch.enable_grad() has no effect outside of no_grad()
        with torch.no_grad():
            if model_name == 'SLENN':
                mm_simis = []
                if 'traj' in modalities:
                    traj_simi, _, _, traj_protos = \
                        model.module.traj_model(inputs['traj'])
                    mm_simis.append(traj_simi) # B n_concept
                if 'ego' in modalities:
                    ego_simi, _, _, ego_protos = \
                        model.module.ego_model(inputs['ego'])
                    mm_simis.append(ego_simi)
                if 'img' in modalities:
                    img_simi, _, _, img_protos = \
                        model.module.img_model(inputs['img'])
                    mm_simis.append(img_simi)
                if 'skeleton' in modalities:
                    sk_simi, _, _, sk_protos = \
                        model.module.sk_model(inputs['skeleton'])
                    mm_simis.append(sk_simi)
                if 'context' in modalities:
                    ctx_simi, _, _, ctx_protos = \
                        model.module.ctx_model(inputs['context'])
                    mm_simis.append(ctx_simi)
                mm_simis = torch.cat(mm_simis, dim=1) # B n_concept
                all_simis.append(mm_simis)
            elif model_name == 'SENN':
                logits, multi_protos, _relevs, recons = model(inputs)
                all_simis.append(multi_protos)
            elif model_name == 'PCPA':
                logits, m_scores = model(inputs)  # m_scores: b,m
                all_simis.append(m_scores)
            else:
                raise NotImplementedError(model_name)
    all_simis = torch.cat(all_simis, dim=0) # n_sample, n_concept
    return all_simis


if __name__ == '__main__':
    test()