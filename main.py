import os
import pickle
import time
from turtle import resizemode
import argparse
import copy
import numpy as np
import pytorch_warmup as warmup

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tools.distributed_parallel import ddp_setup
torch.multiprocessing.set_sharing_strategy('file_system')

from tools.datasets.PIE_JAAD import PIEDataset
from tools.datasets.TITAN import TITAN_dataset
from tools.datasets.nuscenes_dataset import NuscDataset
from tools.datasets.bdd100k import BDD100kDataset

from models.PCPA import PCPA
from models.ped_graph23 import PedGraph
from models.SGNet import SGNet, parse_sgnet_args, parse_sgnet_args2
from models.SGNet_CVAE import SGNet_CVAE
from models.next import Next
from models.deposit import Deposit, deposit_config
from models.PedSpace import PedSpace

from tools.utils import makedir
from tools.log import create_logger
from tools.utils import save_model, seed_all
from tools.visualize.plot import draw_curves2

from train_test import train_test_epoch
from customize_proto import customize_proto
from config import exp_root, dataset_root, cktp_root
from get_args import get_args, process_args
from explain import select_topk
from explain_no_fusion import select_topk_no_fusion

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.mha.set_fastpath_enabled(False)


def get_exp_num():
    f_path = os.path.join(exp_root, 'exp_num.txt')
    if not os.path.exists(f_path):
        exp_num = 0
    else:
        with open(f_path, 'r') as f:
            exp_num = int(f.read().strip())
    exp_num += 1
    with open(f_path, 'w') as f:
        f.write(str(exp_num))
    return exp_num


def construct_data_loader(args, log=print):
    datasets = [
        {
        'train':{k:None for k in args.dataset_names[0]},
        'val':{k:None for k in args.test_dataset_names[0]},
        'test':{k:None for k in args.test_dataset_names[0]},
        },
        {
        'train':{k:None for k in args.dataset_names[1]},
        'val':{k:None for k in args.test_dataset_names[1]},
        'test':{k:None for k in args.test_dataset_names[1]},
        },
    ]
    for stage in range(len(datasets)):
        for subset in datasets[stage]:
            _subset = subset
            _overlap = args.overlap1 if stage == 0 else args.overlap2
            _small_set = args.test_small_set
            if subset == 'pre':
                _subset = 'train'
                _small_set = args.small_set
            elif subset == 'train':
                _small_set = args.small_set
            for name in datasets[stage][subset]:
                if name == 'TITAN':
                    cur_set = TITAN_dataset(sub_set='default_'+_subset, 
                                            offset_traj=False,
                                            img_norm_mode=args.img_norm_mode, 
                                            target_color_order=args.model_color_order,
                                            obs_len=args.obs_len, 
                                            pred_len=args.pred_len, 
                                            overlap_ratio=_overlap, 
                                            obs_fps=args.obs_fps,
                                            recog_act=False,
                                            multi_label_cross=False, 
                                            act_sets=args.act_sets,
                                            loss_weight='sklearn',
                                            small_set=_small_set,
                                            resize_mode=args.resize_mode, 
                                            modalities=args.modalities,
                                            img_format=args.img_format,
                                            sklt_format=args.sklt_format,
                                            ctx_format=args.ctx_format,
                                            traj_format=args.traj_format,
                                            ego_format=args.ego_format,
                                            social_format=args.social_format,
                                            augment_mode=args.augment_mode,
                                            max_n_neighbor=args.max_n_neighbor,
                                            )
                if name in ('PIE', 'JAAD'):
                    cur_set = PIEDataset(dataset_name=name, 
                                         seq_type='crossing',
                                        subset=_subset,
                                        obs_len=args.obs_len, 
                                        pred_len=args.pred_len, 
                                        overlap_ratio=_overlap, 
                                        obs_fps=args.obs_fps,
                                        do_balance=False, 
                                        bbox_size=(224, 224), 
                                        img_norm_mode=args.img_norm_mode, 
                                        target_color_order=args.model_color_order,
                                        resize_mode=args.resize_mode,
                                        modalities=args.modalities,
                                        img_format=args.img_format,
                                        sklt_format=args.sklt_format,
                                        ctx_format=args.ctx_format,
                                        traj_format=args.traj_format,
                                        ego_format=args.ego_format,
                                        social_format=args.social_format,
                                        small_set=_small_set,
                                        tte=args.tte,
                                        recog_act=False,
                                        offset_traj=False,
                                        augment_mode=args.augment_mode,
                                            max_n_neighbor=args.max_n_neighbor,)
                    if subset in ('test', 'val'):
                        cur_set.tte = args.test_tte
                if name == 'nuscenes':
                    cur_set = NuscDataset(subset=_subset,
                                        obs_len=args.obs_len, 
                                        pred_len=args.pred_len, 
                                        overlap_ratio=_overlap, 
                                        obs_fps=args.obs_fps,
                                        small_set=_small_set,
                                        augment_mode=args.augment_mode,
                                        resize_mode=args.resize_mode,
                                        target_color_order=args.model_color_order, 
                                        img_norm_mode=args.img_norm_mode,
                                        modalities=args.modalities,
                                        img_format=args.img_format,
                                        sklt_format=args.sklt_format,
                                        ctx_format=args.ctx_format,
                                        traj_format=args.traj_format,
                                        ego_format=args.ego_format,
                                        social_format=args.social_format,
                                            max_n_neighbor=args.max_n_neighbor,
                                        )
                if name == 'bdd100k':
                    cur_set = BDD100kDataset(subsets=_subset,
                                            obs_len=args.obs_len, 
                                            pred_len=args.pred_len, 
                                            overlap_ratio=_overlap, 
                                            obs_fps=args.obs_fps,
                                            target_color_order=args.model_color_order, 
                                            img_norm_mode=args.img_norm_mode,
                                            small_set=_small_set,
                                            resize_mode=args.resize_mode,
                                            modalities=args.modalities,
                                            img_format=args.img_format,
                                            sklt_format=args.sklt_format,
                                            ctx_format=args.ctx_format,
                                            traj_format=args.traj_format,
                                            ego_format=args.ego_format,
                                            social_format=args.social_format,
                                            augment_mode=args.augment_mode,
                                            max_n_neighbor=args.max_n_neighbor,
                                            )
                datasets[stage][subset][name] = cur_set    
    for stage in range(len(datasets)):
        for _sub in datasets[stage]:
            for nm in datasets[stage][_sub]:
                if datasets[stage][_sub][nm] is not None:
                    log(f'{_sub} {nm} {len(datasets[stage][_sub][nm])}')
    
    concat_train_sets = []
    val_sets = []
    test_sets = []
    train_loaders = []
    val_loaders = []
    test_loaders = []
    for stage in range(len(datasets)):
        concat_train_sets.append(
            torch.utils.data.ConcatDataset(
                [datasets[stage]['train'][k] for k in datasets[stage]['train']]
            )
        )
        val_sets.append([datasets[stage]['val'][k] for k in datasets[stage]['val']])
        test_sets.append([datasets[stage]['test'][k] for k in datasets[stage]['test']])
        _batch_size = args.batch_size[stage]
        train_loaders.append(
            torch.utils.data.DataLoader(concat_train_sets[stage],
                                        batch_size=_batch_size, 
                                        shuffle=args.shuffle,
                                        num_workers=args.dataloader_workers,
                                        pin_memory=True,
                                        drop_last=True)
        )
        val_loaders.append(
            [torch.utils.data.DataLoader(val_sets[stage][i], 
                                        batch_size=_batch_size, 
                                        shuffle=args.shuffle,
                                        num_workers=args.dataloader_workers,
                                        pin_memory=True,
                                        drop_last=True
                                        ) for i in range(len(val_sets[stage]))]
        )
        test_loaders.append(
            [torch.utils.data.DataLoader(test_sets[stage][i], 
                                        batch_size=_batch_size, 
                                        shuffle=args.shuffle,
                                        num_workers=args.dataloader_workers,
                                        pin_memory=True,
                                        drop_last=True
                                        ) for i in range(len(test_sets[stage]))]
        )
    return train_loaders, val_loaders, test_loaders


def construct_model(args, device):
    if args.model_name == 'PCPA':
        model = PCPA(modalities=args.modalities,
                     ctx_bb_nm=args.ctx_backbone_name,
                     proj_norm=args.proj_norm,
                     proj_actv=args.proj_actv,
                     pretrain=False,
                     act_sets=args.act_sets,
                     proj_dim=args.proj_dim,
                     )
    elif args.model_name == 'ped_graph':
        model = PedGraph(modalities=args.modalities,
                         proj_norm=args.proj_norm,
                         proj_actv=args.proj_actv,
                         pretrain=False,
                         act_sets=args.act_sets,
                         n_mlp=1,
                         proj_dim=args.proj_dim,
                         )
    elif args.model_name == 'sgnet':
        sgnet_args = parse_sgnet_args2()
        sgnet_args.enc_steps = args.obs_len
        sgnet_args.dec_steps = args.pred_len
        model = SGNet(sgnet_args)
    elif args.model_name == 'sgnet_cvae':
        sgnet_args = parse_sgnet_args2()
        sgnet_args.enc_steps = args.obs_len
        sgnet_args.dec_steps = args.pred_len
        model = SGNet_CVAE(sgnet_args)
    elif args.model_name == 'next':
        model = Next(obs_len=args.obs_len,
                     pred_len=args.pred_len,
                     action_sets=args.act_sets,
                     )
    elif args.model_name == 'deposit':
        model = Deposit(deposit_config, 
                        device, 
                        target_dim=17*2,
                        modality='sklt')
    elif args.model_name == 'pedspace':
        model = PedSpace(args=args,
                         device=device,
                         )
    return model


def construct_optimizer_scheduler(args, model, train_loaders):
    optimizer1 = None
    optimizer2 = None
    lr_scheduler1 = None
    lr_scheduler2 = None
    if 'sgnet' in args.model_name:
        # Adam lr 5e-5 batch size 128 epoch 50
        optimizer1 = torch.optim.Adam(model.parameters(), 
                                      lr=5e-4, 
                                      weight_decay=5e-4)
        optimizer2 = torch.optim.Adam(model.parameters(),
                                      lr=5e-4, 
                                      weight_decay=5e-4)
        # reg_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(reg_optimizer, factor=0.2, patience=5,
        #                                                         min_lr=1e-10, verbose=1)
        lr_scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 
                                                                  factor=0.2, 
                                                                  patience=5,
                                                                  min_lr=1e-10, 
                                                                  verbose=1)
    elif args.model_name == 'next':
        params = [
            {
                'params': v,
                'lr': 0.1,
                'weight_decay': 0 if 'bias' in k else 1e-4
            } for k, v in model.named_parameters()
        ]
        optimizer1 = torch.optim.Adadelta(params, 
                                            lr=0.1,
                                            weight_decay=5e-4)
        lr_step_gamma = 0.95
        lr_step_size = 2
        lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer1, 
                                                        step_size=lr_step_size, 
                                                        gamma=lr_step_gamma)
    elif args.model_name == 'deposit':
        optimizer1 = torch.optim.Adam(model.parameters(), 
                                      lr=1.0e-3, 
                                      weight_decay=1e-6)
        p1 = int(0.75 * args.epochs1)
        p2 = int(0.9 * args.epochs1)
        lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(
            optimizer1, milestones=[p1, p2], gamma=0.1
        )
    elif args.model_name == 'PCPA':
        optimizer1 = torch.optim.Adam(model.parameters(), 
                                      lr=5e-5, 
                                      weight_decay=1e-3)
    elif args.model_name == 'ped_graph':
        optimizer1 = torch.optim.Adam(model.parameters(), 
                                      lr=5e-5, 
                                      weight_decay=1e-3)
    else:
        backbone_params, other_params = model.get_backbone_params()
        opt_specs1 = [{'params': backbone_params, 'lr': args.backbone_lr1},
                        {'params': other_params, 'lr':args.lr1}]
        opt_specs2 = [{'params': backbone_params, 'lr': args.backbone_lr2},
                        {'params': other_params, 'lr':args.lr2}]
        if args.optim == 'sgd':
            optimizer1 = torch.optim.SGD(opt_specs1, weight_decay=args.weight_decay)
            optimizer2 = torch.optim.SGD(opt_specs2, weight_decay=args.weight_decay)
        elif args.optim == 'adam':
            optimizer1 = torch.optim.Adam(opt_specs1, weight_decay=args.weight_decay, eps=1e-5)
            optimizer2 = torch.optim.Adam(opt_specs2, weight_decay=args.weight_decay, eps=1e-5)
        elif args.optim == 'adamw':
            optimizer1 = torch.optim.AdamW(opt_specs1, weight_decay=args.weight_decay, eps=1e-5)
            optimizer2 = torch.optim.AdamW(opt_specs2, weight_decay=args.weight_decay, eps=1e-5)
        else:
            raise NotImplementedError(args.optim)
        # learning rate scheduler
        if args.epochs1 > 0:
            if args.scheduler_type1 == 'step':
                lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer1, 
                                                                step_size=lr_step_size, 
                                                                gamma=lr_step_gamma)
            elif args.scheduler_type1 == 'cosine':
                lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer1, 
                                                                            T_max=args.t_max, 
                                                                            eta_min=0)
            elif args.scheduler_type1 == 'onecycle':
                lr_scheduler1 = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer1, 
                                                                    max_lr=args.lr1*args.onecycle_div_f, # ?
                                                                    epochs=args.epochs1,
                                                                    steps_per_epoch=len(train_loaders[0]),
                                                                    div_factor=args.onecycle_div_f,
                                                                    )
            else:
                raise NotImplementedError(args.scheduler_type1)
        if args.epochs2 > 0:
            if args.scheduler_type2 == 'step':
                lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer2, 
                                                            step_size=lr_step_size, 
                                                            gamma=lr_step_gamma)
            elif args.scheduler_type2 == 'cosine':
                lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer2, 
                                                                        T_max=args.t_max, 
                                                                        eta_min=0)
            elif args.scheduler_type2 == 'onecycle':
                lr_scheduler2 = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer2, 
                                                                    max_lr=args.lr2*args.onecycle_div_f,
                                                                    epochs=args.epochs2,
                                                                    steps_per_epoch=len(train_loaders[1]),
                                                                    div_factor=args.onecycle_div_f,
                                                                    )
            else:
                raise NotImplementedError(args.scheduler_type2)
    return [optimizer1, optimizer2], [lr_scheduler1, lr_scheduler2]


def update_best_res(best_val_res,
                    best_test_res,
                    curve_dict,
                    test_dataset_names,
                    ):
    if 'cls' in best_val_res:
        for act_set in best_val_res['cls']:
            for metric in best_val_res['cls'][act_set]:
                best_val_res['cls'][act_set][metric] = \
                    sum([curve_dict[d]['val']['cls'][act_set][metric][-1] for d in test_dataset_names]) \
                    / len(test_dataset_names)
                best_test_res['cls'][act_set][metric] = \
                    sum([curve_dict[d]['test']['cls'][act_set][metric][-1] for d in test_dataset_names]) \
                    / len(test_dataset_names)
    for metric in best_val_res:
        if metric == 'cls':
            continue
        best_val_res[metric] = \
            sum([curve_dict[d]['val'][metric][-1] for d in test_dataset_names]) \
            / len(test_dataset_names)
        best_test_res[metric] = \
            sum([curve_dict[d]['test'][metric][-1] for d in test_dataset_names]) \
            / len(test_dataset_names)


def update_res_curve(res_dict, curve_dict, dataset_name, sub_set, plot_dir, plot=False):
    if 'cls' in res_dict:
        for act_set in res_dict['cls']:
            for metric in res_dict['cls'][act_set]:
                curve_dict[dataset_name][sub_set]['cls'][act_set][metric]\
                    .append(res_dict['cls'][act_set][metric])
                if plot:
                    curve_list = [
                                    curve_dict[dataset_name]['val']['cls'][act_set][metric],
                                    curve_dict[dataset_name]['test']['cls'][act_set][metric],
                                    curve_dict['concat']['train']['cls'][act_set][metric],
                                ]
                    draw_curves2(path=os.path.join(plot_dir, 
                                            dataset_name+'_'+act_set+'_'+metric+'.png'), 
                                val_lists=curve_list,
                                labels=['val', 'test', 'train'],
                                colors=['g', 'b', 'r'],
                                vis_every=args.vis_every)
    for metric in res_dict:
        if metric == 'cls':
            continue
        curve_dict[dataset_name][sub_set][metric].append(res_dict[metric])
        if plot:
            curve_list = [
                            curve_dict[dataset_name]['val'][metric],
                            curve_dict[dataset_name]['test'][metric],
                            curve_dict['concat']['train'][metric]
                        ]
            draw_curves2(path=os.path.join(plot_dir, 
                                            dataset_name+'_'+metric+'.png'), 
                        val_lists=curve_list,
                        labels=['val', 'test', 'train'],
                        colors=['g', 'b', 'r'],
                        vis_every=args.vis_every)


def main(rank, world_size, args):
    seed_all(42)
    # device
    local_rank = rank
    ddp = args.ddp and world_size > 1
    # process args
    args = process_args(args)
    # create dirs
    makedir(exp_root)
    exp_id = time.strftime("%d%b%Y-%Hh%Mm%Ss")
    exp_num = get_exp_num()
    model_type = args.model_name
    for m in args.modalities:
        model_type += '_' + m
    exp_dir = os.path.join(exp_root, model_type, f'exp{exp_num}')
    print('Save dir of current exp: ', exp_dir)
    makedir(exp_dir)
    exp_id_f = os.path.join(exp_dir, 'exp_id.txt')
    with open(exp_id_f, 'w') as f:
        f.write(exp_id)
    ckpt_dir = os.path.join(exp_dir, 'ckpt')
    makedir(ckpt_dir)
    plot_dir = os.path.join(exp_dir, 'plot')
    makedir(plot_dir)
    reg_plot_dir = os.path.join(plot_dir, 'reg')
    makedir(reg_plot_dir)
    train_test_plot_dir = os.path.join(plot_dir, 'train_test')
    makedir(train_test_plot_dir)
    # logger
    log, logclose = create_logger(log_filename=os.path.join(exp_dir, 'train.log'))
    log(f'----------------------------Start exp {exp_num}----------------------------')
    log('--------args----------')
    for k in list(vars(args).keys()):
        log(str(k)+': '+str(vars(args)[k]))
    log('--------args----------\n')
    args_dir = os.path.join(exp_dir, 'args.pkl')
    with open(args_dir, 'wb') as f:
        pickle.dump(args, f)
    
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if ddp:
        ddp_setup(local_rank, world_size=torch.cuda.device_count())
        if local_rank != -1:
            torch.cuda.set_device(local_rank)
            device=torch.device("cuda", local_rank)
    
    # load the data
    log('----------------------------Construct data loaders-----------------------------')
    train_loaders, val_loaders, test_loaders = construct_data_loader(args, log)
    # construct the model
    log('----------------------------Construct model-----------------------------')
    model = construct_model(args, device)
    model = model.float().to(device)
    model_parallel = torch.nn.parallel.DataParallel(model)
    # optimizer
    log('----------------------------Construct optimizer----------------------------')
    optimizers, lr_schedulers = construct_optimizer_scheduler(args, model, train_loaders)

    # init curve dict
    metric_dict = {
            'cls': {},
            'traj_mse':[],
            'pose_mse':[],
            'contrast_loss':[],
            'logsig_loss':[],
            'diversity_loss':[],
            'mono_sem_loss': [],
            'mono_sem_l1_loss': [],
            'mono_sem_align_loss': [],
            'batch_sparsity': [],
            'all_sparsity':[],
            'cluster_loss': [],
        }
    for k in args.act_sets:
        metric_dict['cls'][k] = {'acc':[],
                                'auc':[],
                                'f1':[],
                                'map':[],}
    curve_dict_dataset = {
        'train':copy.deepcopy(metric_dict),
        'val':copy.deepcopy(metric_dict),
        'test':copy.deepcopy(metric_dict)
    }
    
    curve_dict = {
        'concat': copy.deepcopy(curve_dict_dataset),
        'TITAN': copy.deepcopy(curve_dict_dataset),
        'PIE': copy.deepcopy(curve_dict_dataset),
        'JAAD': copy.deepcopy(curve_dict_dataset),
        'nuscenes': copy.deepcopy(curve_dict_dataset),
        'bdd100k': copy.deepcopy(curve_dict_dataset),
        # 'macro': copy.deepcopy(curve_dict_dataset),
    }
    
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
    # best res
    best_val_res = {}
    if args.mse_eff[0] > 0:
        best_val_res['traj_mse'] = float('inf')
    if args.pose_mse_eff[0] > 0:
        best_val_res['pose_mse'] = float('inf')
    if args.cls_eff[0] > 0:
        best_val_res['cls'] = {}
        for k in args.act_sets:
            best_val_res['cls'][k] = {'acc':0,
                                    'auc':0,
                                    'f1':0,
                                    'map':0,}
    best_test_res = copy.deepcopy(best_val_res)
    best_epoch_all_test_res = [{d:None for d in args.test_dataset_names[0]}]
    best_epoch_all_test_res.append({d:None for d in args.test_dataset_names[1]})
    cur_epoch_all_test_res = [{d:None for d in args.test_dataset_names[0]}]
    cur_epoch_all_test_res.append({d:None for d in args.test_dataset_names[1]})
    best_epoch_train_res = [{}, {}]
    cur_epoch_train_res = [{}, {}]
    best_e = 0
    # stage 1
    log('----------------------------STAGE 1----------------------------')
    for e in range(1, 1+args.epochs1):
        log(f' stage 1 epoch {e}')
        log('Train')
        train_res = train_test_epoch(args,
                                     model=model_parallel,
                                    model_name=args.model_name,
                                    dataloader=train_loaders[0],
                                    optimizer=optimizers[0],
                                    scheduler=lr_schedulers[0],
                                    log=log,
                                    device=device,
                                    modalities=args.modalities,
                                    loss_params=loss_params[0],
                                    )
        cur_epoch_train_res[0] = train_res
        # add results to curve
        update_res_curve(train_res, curve_dict, 'concat', 'train', train_test_plot_dir)
        # validation and test
        if e%args.test_every == 0:
            log('Val')
            for val_loader in val_loaders[0]:
                cur_dataset = val_loader.dataset.dataset_name
                log(cur_dataset)
                val_res = train_test_epoch(args,
                                           model=model_parallel,
                                            model_name=args.model_name,
                                            dataloader=val_loader,
                                            optimizer=None,
                                            scheduler=lr_schedulers[0],
                                            log=log,
                                            device=device,
                                            modalities=args.modalities,
                                            loss_params=loss_params[0],
                                            )
                update_res_curve(val_res, curve_dict, cur_dataset, 'val', train_test_plot_dir)
            log('Test')
            for test_loader in test_loaders[0]:
                cur_dataset = test_loader.dataset.dataset_name
                log(cur_dataset)
                test_res = train_test_epoch(args,
                                            model=model_parallel,
                                            model_name=args.model_name,
                                            dataloader=test_loader,
                                            optimizer=None,
                                            scheduler=lr_schedulers[0],
                                            log=log,
                                            device=device,
                                            modalities=args.modalities,
                                            loss_params=loss_params[0],
                                            )
                cur_epoch_all_test_res[0][cur_dataset] = test_res
                update_res_curve(test_res, curve_dict, cur_dataset, 'test', train_test_plot_dir, 
                                 plot=True)
            # save best results
            if args.key_metric in ('traj_mse', 'pose_mse') and (args.mse_eff[0] > 0 or args.pose_mse_eff[0] > 0):
                cur_key_res = sum([curve_dict[d]['val'][args.key_metric][-1] for d in args.test_dataset_names[0]]) \
                    / len(args.test_dataset_names[0])
                log(f'cur_key_res: {args.key_metric} {cur_key_res}\n prev best: {best_val_res[args.key_metric]}')
                if cur_key_res < best_val_res[args.key_metric]:
                    try:
                        best_epoch_train_res[0] = cur_epoch_train_res[0]
                        best_epoch_all_test_res[0] = cur_epoch_all_test_res[0]
                        update_best_res(best_val_res, best_test_res, curve_dict, args.test_dataset_names[0])
                    except:
                        import pdb;pdb.set_trace()
                    best_e = e
            elif args.key_metric in ('f1', 'acc', 'auc', 'map') and args.cls_eff[0] > 0:
                try:
                    cur_key_res = sum([curve_dict[d]['val']['cls'][args.key_act_set][args.key_metric][-1] for d in args.test_dataset_names[0]]) \
                    / len(args.test_dataset_names[0])
                    log(f'cur_key_res: {args.key_metric} {cur_key_res}\n prev best: {best_val_res["cls"][args.key_act_set][args.key_metric]}')
                except:
                    import pdb;pdb.set_trace()
                if cur_key_res > best_val_res['cls'][args.key_act_set][args.key_metric]:
                    try:
                        best_epoch_train_res[0] = cur_epoch_train_res[0]
                        best_epoch_all_test_res[0] = cur_epoch_all_test_res[0]
                        update_best_res(best_val_res, best_test_res, curve_dict, args.test_dataset_names[0])
                    except:
                        import pdb;pdb.set_trace()
                    best_e = e
            if local_rank == 0 or not ddp:
                model_path = save_model(model=model, model_dir=ckpt_dir, 
                                        model_name=str(e) + '_',
                                        log=log)
            log(f'bset epoch: {best_e}')
            log(f'current best val results: {best_val_res}')
            log(f'current best test results: {best_test_res}')
            if args.model_name == 'pedspace':
                log(f'train sparsity best epoch: {best_epoch_train_res[0]["all_sparsity"]}')
            log(f'all results of best epoch: {best_epoch_all_test_res[0]}')
        if e%args.explain_every == 0 and args.model_name == 'pedspace':
            log('Selecting topk samples')
            save_root = os.path.join(exp_dir, 'explain', 'stage1_e'+str(e))
            makedir(save_root)
            if args.mm_fusion_mode == 'no_fusion':
                select_topk_no_fusion(dataloader=train_loaders[0], 
                                    model_parallel=model_parallel, 
                                    args=args, 
                                    device=device,
                                    modalities=args.modalities,
                                    save_root=save_root,
                                    log=log)
            else:
                select_topk(dataloader=train_loaders[0], 
                            model_parallel=model_parallel, 
                            args=args, 
                            device=device,
                            modalities=args.modalities,
                            save_root=save_root,
                            log=log)
                        
    log('----------------------------STAGE 2----------------------------')
    # best res
    best_val_res = {}
    if args.mse_eff[1] > 0:
        best_val_res['traj_mse'] = float('inf')
    if args.pose_mse_eff[1] > 0:
        best_val_res['pose_mse'] = float('inf')
    if args.cls_eff[1] > 0:
        best_val_res['cls'] = {}
        for k in args.act_sets:
            best_val_res['cls'][k] = {'acc':0,
                                    'auc':0,
                                    'f1':0,
                                    'map':0,}
    best_test_res = copy.deepcopy(best_val_res)
    best_e = 0
    for e in range(1, 1+args.epochs2):
        log(f' stage 2 epoch {e}')
        train_res = train_test_epoch(args,
                                     model=model_parallel,
                                    model_name=args.model_name,
                                    dataloader=train_loaders[1],
                                    optimizer=optimizers[1],
                                    scheduler=lr_schedulers[1],
                                    log=log,
                                    device=device,
                                    modalities=args.modalities,
                                    loss_params=loss_params[1],
                                    )
        cur_epoch_train_res[1] = train_res
        # add results to curve
        update_res_curve(train_res, curve_dict, 'concat', 'train', train_test_plot_dir)
        # validation and test
        if e%args.test_every == 0:
            log('Val')
            for val_loader in val_loaders[1]:
                cur_dataset = val_loader.dataset.dataset_name
                log(cur_dataset)
                val_res = train_test_epoch(args,
                                            model=model_parallel,
                                            model_name=args.model_name,
                                            dataloader=val_loader,
                                            optimizer=None,
                                            scheduler=lr_schedulers[1],
                                            log=log,
                                            device=device,
                                            modalities=args.modalities,
                                            loss_params=loss_params[1],
                                            )
                update_res_curve(val_res, curve_dict, cur_dataset, 'val', train_test_plot_dir)
            log('Test')
            for test_loader in test_loaders[1]:
                cur_dataset = test_loader.dataset.dataset_name
                log(cur_dataset)
                test_res = train_test_epoch(args,
                                            model=model_parallel,
                                            model_name=args.model_name,
                                            dataloader=test_loader,
                                            optimizer=None,
                                            scheduler=lr_schedulers[1],
                                            log=log,
                                            device=device,
                                            modalities=args.modalities,
                                            loss_params=loss_params[1],
                                            )
                cur_epoch_all_test_res[1][cur_dataset] = test_res
                update_res_curve(test_res, curve_dict, cur_dataset, 'test', train_test_plot_dir,
                                 plot=True)
            # save best results
            if args.key_metric in ('traj_mse', 'pose_mse') and (args.mse_eff[1] > 0 or args.pose_mse_eff[1] > 0):
                cur_key_res = sum([curve_dict[d]['val'][args.key_metric][-1] for d in args.test_dataset_names[1]]) \
                    / len(args.test_dataset_names[1])
                log(f'cur_key_res: {args.key_metric} {cur_key_res}\n prev best: {best_val_res[args.key_metric]}')
                if cur_key_res < best_val_res[args.key_metric]:
                    best_epoch_train_res[1] = cur_epoch_train_res[1]
                    best_epoch_all_test_res[1] = cur_epoch_all_test_res[1]
                    update_best_res(best_val_res, 
                                    best_test_res, 
                                    curve_dict, 
                                    args.test_dataset_names[1])
                    best_e = e
            elif args.key_metric in ('f1', 'acc', 'auc', 'map') and args.cls_eff[1] > 0:
                cur_key_res = sum([curve_dict[d]['val']['cls'][args.key_act_set][args.key_metric][-1] for d in args.test_dataset_names[1]]) \
                    / len(args.test_dataset_names[1])
                log(f'cur_key_res: {args.key_metric} {cur_key_res}\n prev best: {best_val_res["cls"][args.key_act_set][args.key_metric]}')
                if cur_key_res > best_val_res['cls'][args.key_act_set][args.key_metric]:
                    best_epoch_train_res[1] = cur_epoch_train_res[1]
                    best_epoch_all_test_res[1] = cur_epoch_all_test_res[1]
                    update_best_res(best_val_res, 
                                    best_test_res, 
                                    curve_dict, 
                                    args.test_dataset_names[1])
                    best_e = e
            if local_rank == 0 or not ddp:
                model_path = save_model(model=model, model_dir=ckpt_dir, 
                                        model_name=str(e) + '_',
                                        log=log)
            log(f'bset epoch: {best_e}')
            log(f'current best val results: {best_val_res}')
            log(f'current best test results: {best_test_res}')
            if args.model_name == 'pedspace':
                log(f'train sparsity best epoch: {best_epoch_train_res[1]["all_sparsity"]}')
            log(f'all results of best epoch: {best_epoch_all_test_res[1]}')
        # explain
        if e%args.explain_every == 0 and args.model_name == 'pedspace':
            save_root = os.path.join(exp_dir, 'explain', 'stage2_e'+str(e))
            makedir(save_root)
            if args.mm_fusion_mode == 'no_fusion':
                select_topk_no_fusion(dataloader=train_loaders[0], 
                                    model_parallel=model_parallel, 
                                    args=args, 
                                    device=device,
                                    modalities=args.modalities,
                                    save_root=save_root,
                                    log=log)
            else:
                select_topk(dataloader=train_loaders[0], 
                            model_parallel=model_parallel, 
                            args=args, 
                            device=device,
                            modalities=args.modalities,
                            save_root=save_root,
                            log=log)
    if args.model_name == 'pedspace' and args.test_customized_proto:
        log('----------------------------Customize proto----------------------------')
        model = construct_model(args, device)
        model.load_state_dict(torch.load(model_path))
        model = model.float().to(device)
        model_parallel = torch.nn.parallel.DataParallel(model)
        model_parallel.eval()
        model_parallel = customize_proto(args, model_parallel)
        if args.epochs2 > 0:
            final_test_loaders = test_loaders[1]
            final_test_dataset_names = args.test_dataset_names[1]
            final_loss_params = loss_params[1]
        else:
            final_test_loaders = test_loaders[0]
            final_test_dataset_names = args.test_dataset_names[0]
            final_loss_params = loss_params[0]
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

    log(f'Exp {exp_num} finished')                    
    logclose()
    with open(os.path.join(train_test_plot_dir, 'curve_dict.pkl'), 'wb') as f:
        pickle.dump(curve_dict, f)


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    args = get_args()
    if world_size > 1 and args.ddp:
        mp.spawn(main, args=(args),  nprocs=world_size)
    else:
        main(rank=0, world_size=world_size, args=args)
