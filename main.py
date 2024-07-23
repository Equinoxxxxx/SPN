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
from models.SGNet import SGNet, parse_sgnet_args
from models.SGNet_CVAE import SGNet_CVAE

from tools.utils import makedir
from tools.log import create_logger
from tools.utils import save_model, seed_all
from tools.visualize.plot import draw_curves2

from train_test import train_test_epoch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def get_args():
    parser = argparse.ArgumentParser()
    # device
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--ddp", default=False, type=bool)
    # data
    parser.add_argument('--dataset_names1', type=str, default='TITAN')
    parser.add_argument('--test_dataset_names1', type=str, default='TTIAN')
    parser.add_argument('--dataset_names2', type=str, default='TITAN')
    parser.add_argument('--test_dataset_names2', type=str, default='TITAN')
    parser.add_argument('--small_set1', type=float, default=0)
    parser.add_argument('--small_set2', type=float, default=0)
    parser.add_argument('--test_small_set', type=float, default=0)
    parser.add_argument('--obs_len', type=int, default=4)
    parser.add_argument('--pred_len', type=int, default=4)
    parser.add_argument('--obs_fps', type=int, default=2)
    parser.add_argument('--apply_tte', type=int, default=1)
    parser.add_argument('--test_apply_tte', type=int, default=1)
    parser.add_argument('--augment_mode', type=str, default='none')
    parser.add_argument('--img_norm_mode', type=str, default='torch')
    parser.add_argument('--color_order', type=str, default='BGR')
    parser.add_argument('--resize_mode', type=str, default='even_padded')
    parser.add_argument('--overlap1', type=float, default=0.5)
    parser.add_argument('--overlap2', type=float, default=0.5)
    parser.add_argument('--test_overlap', type=float, default=0.5)
    parser.add_argument('--dataloader_workers', type=int, default=8)
    parser.add_argument('--shuffle', type=int, default=1)

    # train
    parser.add_argument('--reg_first', type=int, default=1)
    parser.add_argument('--epochs1', type=int, default=50)
    parser.add_argument('--epochs2', type=int, default=50)
    parser.add_argument('--warm_step1', type=int, default=1)
    parser.add_argument('--batch_size1', type=int, default=128)
    parser.add_argument('--batch_size2', type=int, default=128)
    parser.add_argument('--test_every', type=int, default=2)
    parser.add_argument('--explain_every', type=int, default=10)
    parser.add_argument('--vis_every', type=int, default=2)
    parser.add_argument('--lr1', type=float, default=0.001)
    parser.add_argument('--lr2', type=float, default=0.001)
    parser.add_argument('--backbone_lr1', type=float, default=0.001)
    parser.add_argument('--backbone_lr2', type=float, default=0.001)
    parser.add_argument('--scheduler1', type=str, default='onecycle')
    parser.add_argument('--scheduler2', type=str, default='onecycle')
    parser.add_argument('--onecycle_div_f', type=int, default=10)
    parser.add_argument('--batch_schedule', type=int, default=0)
    parser.add_argument('--lr_step_size', type=int, default=5)
    parser.add_argument('--lr_step_gamma', type=float, default=1.)
    parser.add_argument('--t_max', type=float, default=10)
    parser.add_argument('--optim', type=str, default='adamw')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--act_sets', type=str, default='cross')
    parser.add_argument('--key_metric', type=str, default='f1')
    parser.add_argument('--key_act_set', type=str, default='cross')

    # loss
    parser.add_argument('--mse_eff1', type=float, default=0.5)
    parser.add_argument('--mse_eff2', type=float, default=0.1)
    parser.add_argument('--pose_mse_eff1', type=float, default=0.5)
    parser.add_argument('--pose_mse_eff2', type=float, default=0.1)
    parser.add_argument('--cls_loss', type=str, default='weighted_ce')
    parser.add_argument('--cls_eff1', type=float, default=0)
    parser.add_argument('--cls_eff2', type=float, default=1.)
    parser.add_argument('--logsig_thresh', type=float, default=100)
    parser.add_argument('--logsig_loss_eff', type=float, default=0.1)
    parser.add_argument('--logsig_loss', type=str, default='kl')

    # model
    parser.add_argument('--pretrain_mode', type=str, default='contrast')
    parser.add_argument('--model_name', type=str, default='sgnet')
    parser.add_argument('--pair_mode', type=str, default='pair_wise')
    parser.add_argument('--simi_func', type=str, default='dot_prod')
    parser.add_argument('--bridge_m', type=str, default='sk')
    parser.add_argument('--n_proto', type=int, default=50)
    parser.add_argument('--proj_dim', type=int, default=0)
    parser.add_argument('--pool', type=str, default='max')
    parser.add_argument('--n_layer_proj', type=int, default=3)
    parser.add_argument('--proj_norm', type=str, default='ln')
    parser.add_argument('--proj_actv', type=str, default='leakyrelu')
    parser.add_argument('--uncertainty', type=str, default='none')
    parser.add_argument('--n_pred_sampling', type=int, default=3)

    # modality
    parser.add_argument('--modalities', type=str, default='sklt_ctx_traj_ego')
    # img settingf
    parser.add_argument('--img_format', type=str, default='')
    parser.add_argument('--img_backbone_name', type=str, default='R3D18')
    # sk setting
    parser.add_argument('--sklt_format', type=str, default='0-1coord')
    parser.add_argument('--sklt_backbone_name', type=str, default='poseC3D')
    # ctx setting
    parser.add_argument('--ctx_format', type=str, default='ori_local')
    parser.add_argument('--seg_cls', type=str, default='person,vehicles,roads,traffic_lights')
    parser.add_argument('--fuse_mode', type=str, default='transformer')
    parser.add_argument('--ctx_backbone_name', type=str, default='C3D_t4')
    # traj setting
    parser.add_argument('--traj_format', type=str, default='0-1ltrb')
    parser.add_argument('--traj_backbone_name', type=str, default='lstm')
    # ego setting
    parser.add_argument('--ego_format', type=str, default='accel')
    parser.add_argument('--ego_backbone_name', type=str, default='lstm')

    args = parser.parse_args()

    return args

def main(rank, world_size, args):
    seed_all(42)
    # device
    local_rank = rank
    ddp = args.ddp and world_size > 1
    # data
    dataset_names1 = args.dataset_names1.split('_')
    dataset_names2 = args.dataset_names2.split('_')
    test_dataset_names1 = args.test_dataset_names1.split('_')
    test_dataset_names2 = args.test_dataset_names2.split('_')
    small_set1 = args.small_set1
    small_set2 = args.small_set2
    test_small_set = args.test_small_set
    obs_len = args.obs_len
    pred_len = args.pred_len
    obs_fps = args.obs_fps
    tte = None
    test_tte = None
    apply_tte = args.apply_tte
    test_apply_tte = args.test_apply_tte
    if apply_tte:
        tte = [0, int((obs_len+pred_len+1)/obs_fps*30)]  # before donwsample
    if test_apply_tte:
        test_tte = [0, int((obs_len+pred_len+1)/obs_fps*30)]  # before donwsample
    augment_mode = args.augment_mode
    img_norm_mode = args.img_norm_mode
    color_order = args.color_order
    resize_mode = args.resize_mode
    overlap1 = args.overlap1
    overlap2 = args.overlap2
    dataloader_workers = args.dataloader_workers
    shuffle = args.shuffle
    # train
    reg_first = args.reg_first
    epochs1 = args.epochs1
    epochs2 = args.epochs2
    warm_step1 = args.warm_step1
    batch_size1 = args.batch_size1
    batch_size2 = args.batch_size2
    test_every = args.test_every
    explain_every = args.explain_every
    vis_every = args.vis_every
    lr1 = args.lr1
    lr2 = args.lr2
    backbone_lr1 = args.backbone_lr1
    backbone_lr2 = args.backbone_lr2
    scheduler1 = args.scheduler1
    scheduler2 = args.scheduler2
    onecycle_div_f = args.onecycle_div_f
    batch_schedule = args.batch_schedule
    lr_step_size = args.lr_step_size
    lr_step_gamma = args.lr_step_gamma
    t_max = args.t_max
    optim = args.optim
    wd = args.weight_decay
    act_sets = args.act_sets.split('_')
    key_metric = args.key_metric
    key_act_set = args.key_act_set
    if len(act_sets) == 1:
        key_act_set = act_sets[0]
    assert key_act_set in act_sets + ['macro']

    # loss
    mse_eff1 = args.mse_eff1
    mse_eff2 = args.mse_eff2
    pose_mse_eff1 = args.pose_mse_eff1
    pose_mse_eff2 = args.pose_mse_eff2
    cls_loss = args.cls_loss
    cls_eff1 = args.cls_eff1
    cls_eff2 = args.cls_eff2
    logsig_thresh = args.logsig_thresh
    logsig_loss_eff = args.logsig_loss_eff
    logsig_loss_func = args.logsig_loss
    # model
    model_name = args.model_name
    simi_func = args.simi_func
    pair_mode = args.pair_mode
    bridge_m = args.bridge_m
    n_proto = args.n_proto
    proj_dim = args.proj_dim
    pool = args.pool
    n_layer_proj = args.n_layer_proj
    proj_norm = args.proj_norm
    proj_actv = args.proj_actv
    uncertainty = args.uncertainty
    n_pred_sampling = args.n_pred_sampling
    n_mlp = args.n_mlp
    # modality
    modalities = args.modalities.split('_')
    img_format = args.img_format
    img_backbone_name = args.img_backbone_name
    sklt_format = args.sklt_format
    sk_backbone_name = args.sklt_backbone_name
    ctx_format = args.ctx_format
    seg_cls = args.seg_cls
    fuse_mode = args.fuse_mode
    ctx_backbone_name = args.ctx_backbone_name
    traj_format = args.traj_format
    traj_backbone_name = args.traj_backbone_name
    ego_format = args.ego_format
    ego_backbone_name = args.ego_backbone_name

    # conditioned config
    if model_name == 'PCPA':
        if 'JAAD' in test_dataset_names2 or 'JAAD' in test_dataset_names1:
            modalities = ['sklt','ctx', 'traj']
        else:
            modalities = ['sklt','ctx', 'traj', 'ego']
        sklt_format = 'coord'
        if '0-1' in sklt_format:
            sklt_format = '0-1coord'
        ctx_format = 'ori_local'
    elif model_name == 'ped_graph':
        if 'JAAD' in test_dataset_names2 or 'JAAD' in test_dataset_names1:
            modalities = ['sklt','ctx']
        else:
            modalities = ['sklt','ctx', 'ego']
        sklt_format = 'coord'
        if '0-1' in sklt_format:
            sklt_format = '0-1coord'
        ctx_format = 'ped_graph'

    
    if model_name in ('sgnet', 'sgnet_cvae', 'deposit'):
        cls_eff2 = 0

    if 'R3D' in img_backbone_name or 'csn' in img_backbone_name\
        or 'R3D' in ctx_backbone_name or 'csn' in ctx_backbone_name:
        img_norm_mode = 'kinetics'
    if img_norm_mode in ('kinetics', '0.5', 'activitynet'):
        color_order = 'RGB'
    else:
        color_order = 'BGR'
    
    if uncertainty != 'gaussian':
        logsig_loss_eff = 0
    
    # create dirs
    exp_id = time.strftime("%d%b%Y-%Hh%Mm%Ss")
    work_dir = '../work_dirs/exp/contrast'
    makedir(work_dir)
    model_type = model_name
    for m in modalities:
        model_type += '_' + m
    model_dir = os.path.join(work_dir, model_type, exp_id)
    print('Save dir of current exp: ', model_dir)
    makedir(model_dir)
    ckpt_dir = os.path.join(model_dir, 'ckpt')
    makedir(ckpt_dir)
    plot_dir = os.path.join(model_dir, 'plot')
    makedir(plot_dir)
    reg_plot_dir = os.path.join(plot_dir, 'reg')
    makedir(reg_plot_dir)
    train_test_plot_dir = os.path.join(plot_dir, 'train_test')
    makedir(train_test_plot_dir)
    # logger
    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    log('--------args----------')
    for k in list(vars(args).keys()):
        log(str(k)+': '+str(vars(args)[k]))
    log('--------args----------\n')
    args_dir = os.path.join(model_dir, 'args.pkl')
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
    log('----------------------------Load data-----------------------------')
    reg_datasets = {
        'train':{k:None for k in dataset_names1},
        'val':{k:None for k in dataset_names1},
        'test':{k:None for k in dataset_names1},
    }
    cls_datasets = {
        'train':{k:None for k in dataset_names1},
        'val':{k:None for k in dataset_names1},
        'test':{k:None for k in dataset_names1},
    }
    datasets = {
        'reg': reg_datasets,
        'cls': cls_datasets,
    }
    for task in datasets:
        for subset in datasets:
            _subset = subset
            _overlap = overlap1
            _small_set = test_small_set
            if subset == 'pre':
                _subset = 'train'
                _small_set = small_set1
            elif subset == 'train':
                _small_set = small_set1
            for name in datasets[task][subset]:
                if name == 'TITAN':
                    cur_set = TITAN_dataset(sub_set='default_'+_subset, 
                                            norm_traj=False,
                                            img_norm_mode=img_norm_mode, color_order=color_order,
                                            obs_len=obs_len, pred_len=pred_len, overlap_ratio=_overlap, 
                                            obs_fps=obs_fps,
                                            recog_act=False,
                                            multi_label_cross=False, 
                                            use_atomic=False, 
                                            use_complex=False, 
                                            use_communicative=False, 
                                            use_transporting=False, 
                                            use_age=False,
                                            loss_weight='sklearn',
                                            small_set=_small_set,
                                            resize_mode=resize_mode, 
                                            modalities=modalities,
                                            img_format=img_format,
                                            sklt_format=sklt_format,
                                            ctx_format=ctx_format,
                                            traj_format=traj_format,
                                            ego_format=ego_format,
                                            augment_mode=augment_mode,
                                            )
                if name in ('PIE', 'JAAD'):
                    cur_set = PIEDataset(dataset_name=name, seq_type='crossing',
                                        subset=_subset,
                                        obs_len=obs_len, pred_len=pred_len, overlap_ratio=_overlap, 
                                        obs_fps=obs_fps,
                                        do_balance=False, 
                                        bbox_size=(224, 224), 
                                        img_norm_mode=img_norm_mode, color_order=color_order,
                                        resize_mode=resize_mode,
                                        modalities=modalities,
                                        img_format=img_format,
                                        sklt_format=sklt_format,
                                        ctx_format=ctx_format,
                                        traj_format=traj_format,
                                        ego_format=ego_format,
                                        small_set=_small_set,
                                        tte=tte,
                                        recog_act=False,
                                        normalize_pos=False,
                                        augment_mode=augment_mode)
                    if subset in ('test', 'val'):
                        cur_set.tte = test_tte
                if name == 'nuscenes':
                    cur_set = NuscDataset(subset=_subset,
                                        obs_len=obs_len, pred_len=pred_len, overlap_ratio=_overlap, 
                                        obs_fps=obs_fps,
                                        small_set=_small_set,
                                        augment_mode=augment_mode,
                                        resize_mode=resize_mode,
                                        color_order=color_order, img_norm_mode=img_norm_mode,
                                        modalities=modalities,
                                        img_format=img_format,
                                        sklt_format=sklt_format,
                                        ctx_format=ctx_format,
                                        traj_format=traj_format,
                                        ego_format=ego_format
                                        )
                if name == 'bdd100k':
                    cur_set = BDD100kDataset(subsets=_subset,
                                            obs_len=obs_len, pred_len=pred_len, overlap_ratio=_overlap, 
                                            obs_fps=obs_fps,
                                            color_order=color_order, img_norm_mode=img_norm_mode,
                                            small_set=_small_set,
                                            resize_mode=resize_mode,
                                            modalities=modalities,
                                            img_format=img_format,
                                            sklt_format=sklt_format,
                                            ctx_format=ctx_format,
                                            traj_format=traj_format,
                                            ego_format=ego_format,
                                            augment_mode=augment_mode
                                            )
                datasets[subset][name] = cur_set    
    for task in datasets:
        for _sub in datasets:
            for nm in datasets[_sub]:
                if datasets[_sub][nm] is not None:
                    log(f'{_sub} {nm} {len(datasets[_sub][nm])}')
    
    reg_train_set = torch.utils.data.ConcatDataset(
        [datasets['reg']['train'][k] for k in datasets['reg']['train']]
        )
    cls_train_set = torch.utils.data.ConcatDataset(
        [datasets['cls']['train'][k] for k in datasets['cls']['train']]
        )

    reg_val_sets = [datasets['reg']['val'][k] for k in datasets['reg']['val']]
    cls_val_sets = [datasets['cls']['val'][k] for k in datasets['cls']['val']]

    reg_test_sets = [datasets['reg']['test'][k] for k in datasets['reg']['test']]
    cls_test_sets = [datasets['cls']['test'][k] for k in datasets['cls']['test']]

    reg_train_loader = torch.utils.data.DataLoader(reg_train_set, 
                                             batch_size=batch_size1, 
                                             shuffle=shuffle,
                                             num_workers=dataloader_workers,
                                             pin_memory=True)
    cls_train_loader = torch.utils.data.DataLoader(cls_train_set, 
                                             batch_size=batch_size1, 
                                             shuffle=shuffle,
                                             num_workers=dataloader_workers,
                                             pin_memory=True)
    reg_val_loaders = [torch.utils.data.DataLoader(cur_set, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle,
                                             num_workers=dataloader_workers,
                                             pin_memory=True
                                             ) for cur_set in reg_val_sets]
    cls_val_loaders = [torch.utils.data.DataLoader(cur_set, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle,
                                             num_workers=dataloader_workers,
                                             pin_memory=True
                                             ) for cur_set in cls_val_sets]
    reg_test_loaders = [torch.utils.data.DataLoader(cur_set, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle,
                                             num_workers=dataloader_workers,
                                             pin_memory=True
                                             ) for cur_set in reg_test_sets]
    cls_test_loaders = [torch.utils.data.DataLoader(cur_set, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle,
                                             num_workers=dataloader_workers,
                                             pin_memory=True
                                             ) for cur_set in cls_test_sets]

    # construct the model
    log('----------------------------Construct model-----------------------------')
    if model_name == 'PCPA':
        model = PCPA(modalities=modalities,
                     ctx_bb_nm=ctx_backbone_name,
                     proj_norm=proj_norm,
                     proj_actv=proj_actv,
                     pretrain=True,
                     act_sets=act_sets,
                     n_mlp=n_mlp,
                     proj_dim=proj_dim,
                     )
    elif model_name == 'ped_graph':
        model = PedGraph(modalities=modalities,
                         proj_norm=proj_norm,
                         proj_actv=proj_actv,
                         pretrain=True,
                         act_sets=act_sets,
                         n_mlp=n_mlp,
                         proj_dim=proj_dim,
                         )
    elif model_name == 'sgnet':
        sgnet_args = parse_sgnet_args()
        model = SGNet(sgnet_args)
    elif model_name == 'sgnet_cvae':
        sgnet_args = parse_sgnet_args()
        model = SGNet_CVAE(sgnet_args)
    elif model_name == 'next':
        raise NotImplementedError()
    elif model_name == 'deposit':
        raise NotImplementedError()
    else:
        raise NotImplementedError
    model = model.float().to(device)
    model_parallel = torch.nn.parallel.DataParallel(model)
    log('----------------------------Construct optimizer-----------------------------')
    # optimizer
    reg_lr_scheduler = None
    if 'sgnet' in model_name:
        # Adam lr 5e-5 batch size 128 epoch 50
        reg_optimizer = torch.optim.Adam(model.parameters(), lr=sgnet_args.lr)
        # reg_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(reg_optimizer, factor=0.2, patience=5,
        #                                                         min_lr=1e-10, verbose=1)
    else:
        backbone_params, other_params = model.get_pretrain_params()
        reg_opt_specs = [{'params': backbone_params, 'lr': backbone_lr1},
                        {'params': other_params, 'lr':lr1}]
        opt_specs = [{'params': backbone_params, 'lr': backbone_lr2},
                        {'params': other_params, 'lr':lr2}]
        
        if optim == 'sgd':
            reg_optimizer = torch.optim.SGD(reg_opt_specs, lr=backbone_lr1, weight_decay=wd)
            optimizer = torch.optim.SGD(opt_specs, lr=lr2, weight_decay=wd)
        elif optim == 'adam':
            reg_optimizer = torch.optim.Adam(reg_opt_specs, lr=backbone_lr1, weight_decay=wd, eps=1e-5)
            optimizer = torch.optim.Adam(opt_specs, lr=lr2, weight_decay=wd, eps=1e-5)
        elif optim == 'adamw':
            reg_optimizer = torch.optim.AdamW(reg_opt_specs, lr=backbone_lr1, weight_decay=wd, eps=1e-5)
            optimizer = torch.optim.AdamW(opt_specs, lr=lr2, weight_decay=wd, eps=1e-5)
        else:
            raise NotImplementedError(optim)
        
        # learning rate scheduler
        if scheduler1 == 'step':
            reg_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=reg_optimizer, 
                                                            step_size=lr_step_size, 
                                                            gamma=lr_step_gamma)
        elif scheduler1 == 'cosine':
            reg_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=reg_optimizer, 
                                                                        T_max=t_max, 
                                                                        eta_min=0)
        elif scheduler1 == 'onecycle':
            if epochs1 == 0:
                reg_lr_scheduler = None
            else:
                reg_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=reg_optimizer, 
                                                                    max_lr=backbone_lr1*onecycle_div_f,
                                                                    epochs=epochs1,
                                                                    steps_per_epoch=len(reg_train_loader),
                                                                    div_factor=onecycle_div_f,
                                                                    )
        else:
            raise NotImplementedError(scheduler1)
        p_warmer = warmup.LinearWarmup(reg_optimizer, 
                                    warmup_period=warm_step1)
        if scheduler2 == 'step':
            cls_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                        step_size=lr_step_size, 
                                                        gamma=lr_step_gamma)
        elif scheduler2 == 'cosine':
            cls_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                                                    T_max=t_max, 
                                                                    eta_min=0)
        elif scheduler2 == 'onecycle':
            cls_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, 
                                                                max_lr=lr2*10,
                                                                epochs=epochs,
                                                                steps_per_epoch=len(cls_train_loader),
                                                                div_factor=10,
                                                                )
        else:
            raise NotImplementedError(scheduler2)

    # init curve dict
    metric_dict = {
            'acc':[],
            'auc':[],
            'f1':[],
            'map':[],
            'logsig_loss':[],
            'traj_mse':[],
            'pose_mse':[]
        }
    curve_dict_dataset = {
        'train':copy.deepcopy(metric_dict),
        'val':copy.deepcopy(metric_dict),
        'test':copy.deepcopy(metric_dict)
    }
    
    curve_dict = {
        'TITAN': curve_dict_dataset,
        'PIE': copy.deepcopy(curve_dict_dataset),
        'JAAD': copy.deepcopy(curve_dict_dataset),
        'nuscenes': copy.deepcopy(curve_dict_dataset),
        'bdd100k': copy.deepcopy(curve_dict_dataset),
        # 'macro': copy.deepcopy(curve_dict_dataset),
    }

    
    # loss params
    reg_loss_params = {
        'mse_eff': mse_eff1,
        'pose_mse_eff': pose_mse_eff1,
        'cls_eff':0,
        'cls_loss': cls_loss,
        'logsig_loss_eff':logsig_loss_eff,
        'logsig_thresh':logsig_thresh,
    }
    cls_loss_params = {
        'mse_eff': mse_eff2,
        'pose_mse_eff': pose_mse_eff2,
        'cls_eff':cls_eff2,
        'cls_loss': cls_loss,
        'logsig_loss_eff':logsig_loss_eff,
        'logsig_thresh':logsig_thresh,
    }

    # start training
    if reg_first:
        # regression task
        for e in range(1, 1+epochs1):
            log(f'Reg {e} epoch')
            train_res = train_test_epoch(model_parallel,
                                         model_name,
                                         reg_train_loader,
                                         optimizer=)
