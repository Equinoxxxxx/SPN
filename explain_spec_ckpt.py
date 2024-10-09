import torch
import numpy as np
from tqdm import tqdm
import os
import cv2
import copy
import argparse
import os
import pickle

from tools.utils import makedir, write_info_txt
from main import construct_data_loader, construct_model
from tools.log import create_logger
from explain import select_topk
from explain_no_fusion import select_topk_no_fusion
from train_test import train_test_epoch
from customize_proto import customize_proto


def explain_spec_ckpt():
    parser = argparse.ArgumentParser(description='explain args')
    parser.add_argument('--args_path', type=str, 
                        default='/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp434/args.pkl')
    parser.add_argument('--ckpt_path', type=str, 
                        default='/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp434/ckpt/28_0.0000.pth')
    parser.add_argument('--proto_value_to_rank', type=str, default='sparsity')
    parser.add_argument('--proto_rank_criteria', type=str, default='num_select')
    parser.add_argument('--proto_num_select', type=int, default=5)
    parser.add_argument('--do_explain', type=int, default=1)
    explain_args = parser.parse_args()

    # load args
    with open(explain_args.args_path, 'rb') as f:
        args = pickle.load(f)
    args.test_customized_proto = True
    args.proto_value_to_rank = explain_args.proto_value_to_rank
    args.proto_rank_criteria = explain_args.proto_rank_criteria
    args.proto_num_select = explain_args.proto_num_select
    if not hasattr(args, 'max_n_neighbor'):
        args.max_n_neighbor = 10
    # log
    ckpt_epoch = int(explain_args.ckpt_path.split('/')[-1].split('_')[0])
    exp_dir = explain_args.args_path.replace('args.pkl', '')
    if explain_args.do_explain:
        explain_root = os.path.join(exp_dir, f'explain_epoch_{ckpt_epoch}' )
    else:
        explain_root = os.path.join(exp_dir, f'test_epoch_{ckpt_epoch}' )
    makedir(explain_root)
    log, logclose = create_logger(log_filename=os.path.join(explain_root, 'explain.log'))
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load the data
    log('----------------------------Construct data loaders-----------------------------')
    train_loaders, val_loaders, test_loaders = construct_data_loader(args, log)
    # construct the model
    log('----------------------------Construct model-----------------------------')
    model = construct_model(args, device)
    model = model.float().to(device)
    state_dict = torch.load(explain_args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model_parallel = torch.nn.parallel.DataParallel(model)
    model_parallel.eval()
    if explain_args.do_explain:
        log('----------------------------Explain-----------------------------')
        if args.mm_fusion_mode == 'no_fusion':
            select_topk_no_fusion(dataloader=train_loaders[0], 
                                model_parallel=model_parallel, 
                                args=args, 
                                device=device,
                                modalities=args.modalities,
                                save_root=explain_root,
                                log=log)
        else:
            select_topk(dataloader=train_loaders[0], 
                        model_parallel=model_parallel, 
                        args=args, 
                        device=device,
                        modalities=args.modalities,
                        save_root=explain_root,
                        log=log)
    
    # loss params
    loss_params = [{
        'mse_eff': args.mse_eff[0],
        'pose_mse_eff': args.pose_mse_eff[0],
        'stoch_mse_type': args.stoch_mse_type,
        'n_sampling': args.n_pred_sampling,
        'cls_eff':args.cls_eff[0],
        'cls_loss_func': args.cls_loss_func,
        'logsig_loss_func': args.logsig_loss_func,
        'logsig_loss_eff':args.logsig_loss_eff,
        'logsig_thresh':args.logsig_thresh,
        'diversity_loss_func': args.diversity_loss_func,
        'diversity_loss_eff': args.diversity_loss_eff,
        'mono_sem_eff': args.mono_sem_eff,
        'mono_sem_l1_eff': args.mono_sem_l1_eff,
        'mono_sem_align_func': args.mono_sem_align_func,
        'mono_sem_align_eff': args.mono_sem_align_eff,
        'cluster_loss_eff': args.cluster_loss_eff,
    },
    {
        'mse_eff': args.mse_eff[1],
        'pose_mse_eff': args.pose_mse_eff[1],
        'stoch_mse_type': args.stoch_mse_type,
        'n_sampling': args.n_pred_sampling,
        'cls_eff':args.cls_eff[1],
        'cls_loss_func': args.cls_loss_func,
        'logsig_loss_func': args.logsig_loss_func,
        'logsig_loss_eff': args.logsig_loss_eff,
        'logsig_thresh': args.logsig_thresh,
        'diversity_loss_func': args.diversity_loss_func,
        'diversity_loss_eff': args.diversity_loss_eff,
        'mono_sem_eff': args.mono_sem_eff,
        'mono_sem_l1_eff': args.mono_sem_l1_eff,
        'mono_sem_align_func': args.mono_sem_align_func,
        'mono_sem_align_eff': args.mono_sem_align_eff,
        'cluster_loss_eff': args.cluster_loss_eff,
    }]
    if args.model_name == 'pedspace' and args.test_customized_proto:
        log('----------------------------Customize proto----------------------------')
        model = construct_model(args, device)
        model.load_state_dict(torch.load(explain_args.ckpt_path))
        model = model.float().to(device)
        model_parallel = torch.nn.parallel.DataParallel(model)
        model_parallel.eval()
        if args.epochs2 > 0:
            final_test_loaders = test_loaders[1]
            final_test_dataset_names = args.test_dataset_names[1]
            final_loss_params = loss_params[1]
        else:
            final_test_loaders = test_loaders[0]
            final_test_dataset_names = args.test_dataset_names[0]
            final_loss_params = loss_params[0]
        # get sparsity
        if explain_args.proto_value_to_rank == 'sparsity' and model.all_sparsity is None:
            log('Get sparsity')
            _ = train_test_epoch(args,
                                model=model_parallel,
                                model_name=args.model_name,
                                dataloader=train_loaders[0],
                                optimizer=None,
                                scheduler=None,
                                log=log,
                                device=device,
                                modalities=args.modalities,
                                loss_params=final_loss_params,
                                )
        model_parallel = customize_proto(args, model_parallel)
        
        customize_proto_res = {d:None for d in final_test_dataset_names}
        for test_loader in final_test_loaders:
            cur_dataset = test_loader.dataset.dataset_name
            log(cur_dataset)
            test_res = train_test_epoch(args,
                                        model=model_parallel,
                                        model_name=args.model_name,
                                        dataloader=test_loader,
                                        optimizer=None,
                                        scheduler=None,
                                        log=log,
                                        device=device,
                                        modalities=args.modalities,
                                        loss_params=final_loss_params,
                                        )
            customize_proto_res[cur_dataset] = test_res
        log(f'Customize proto results\n  {customize_proto_res}')
    logclose()

if __name__ == '__main__':
    explain_spec_ckpt()