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
from models.next import Next

from tools.utils import makedir
from tools.log import create_logger
from tools.utils import save_model, seed_all
from tools.visualize.plot import draw_curves2

from train_test import train_test_epoch
from config import exp_root, dataset_root, cktp_root

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
    parser.add_argument('--small_set', type=float, default=0)
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
    parser.add_argument('--scheduler_type1', type=str, default='onecycle')
    parser.add_argument('--scheduler_type2', type=str, default='onecycle')
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
    parser.add_argument('--cls_loss_func', type=str, default='weighted_ce')
    parser.add_argument('--cls_eff1', type=float, default=0)
    parser.add_argument('--cls_eff2', type=float, default=1.)
    parser.add_argument('--logsig_thresh', type=float, default=100)
    parser.add_argument('--logsig_loss_eff', type=float, default=0.1)
    parser.add_argument('--logsig_loss_func', type=str, default='kl')
    parser.add_argument('--mono_sem_eff', type=float, default=0)
    parser.add_argument('--stoch_mse_type', type=str, default='avg')

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
    parser.add_argument('--n_pred_sampling', type=int, default=5)

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
    # social setting
    parser.add_argument('--social_backbone_name', type=str, default='transformer')
    # traj setting
    parser.add_argument('--traj_format', type=str, default='0-1ltrb')
    parser.add_argument('--traj_backbone_name', type=str, default='lstm')
    # ego setting
    parser.add_argument('--ego_format', type=str, default='accel')
    parser.add_argument('--ego_backbone_name', type=str, default='lstm')

    args = parser.parse_args()

    return args

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

def process_args(args):
    args.dataset_names = [args.dataset_names1.split('_'),
                          args.dataset_names2.split('_')]
    args.test_dataset_names = [args.test_dataset_names1.split('_'),
                               args.test_dataset_names2.split('_')]
    args.act_sets = args.act_sets.split('_')
    args.modalities = args.modalities.split('_')
    args.tte = None
    args.test_tte = None
    if args.apply_tte:
        args.tte = [0, int((args.obs_len + args.pred_len + 1) / args.obs_fps * 30)]  # before downsample
    if args.test_apply_tte:
        args.test_tte = [0, int((args.obs_len + args.pred_len + 1) / args.obs_fps * 30)]  # before downsample
    # conditioned config
    if args.model_name in ('sgnet', 'sgnet_cvae', 'deposit'):
        args.cls_eff1 = 0
        args.cls_eff2 = 0
        args.mse_eff1 = 1
        args.traj_format = '0-1ltrb'
    if args.model_name == 'PCPA':
        args.mse_eff1 = 0
        args.mse_eff2 = 0
        if 'JAAD' in args.test_dataset_names2 or 'JAAD' in args.test_dataset_names1:
            args.modalities = ['sklt','ctx', 'traj']
        else:
            args.modalities = ['sklt','ctx', 'traj', 'ego']
        if '0-1' in args.sklt_format:
            args.sklt_format = '0-1coord'
        else:
            args.sklt_format = 'coord'
        ctx_format = 'ori_local'
    elif args.model_name == 'ped_graph':
        args.mse_eff1 = 0
        args.mse_eff2 = 0
        if 'JAAD' in args.test_dataset_names2 or 'JAAD' in args.test_dataset_names1:
            args.modalities = ['sklt','ctx']
        else:
            args.modalities = ['sklt','ctx', 'ego']
        if '0-1' in args.sklt_format:
            args.sklt_format = '0-1coord'
        else:
            args.sklt_format = 'coord'
        args.ctx_format = 'ped_graph'
    elif args.model_name == 'next':
        args.cls_eff1 = 1
        args.mse_eff1 = 1
        args.epochs1 = 100
        args.epochs2 = 0
        args.batch_size1 = 64
        args.modalities = ['img', 'sklt', 'ctx', 'social', 'traj']
        args.sklt_format = '0-1coord'
        args.ctx_format = 'ped_graph'
    elif args.model_name in ('sgnet', 'sgnet_cvae'):
        args.epochs1 = 50
        args.epochs2 = 0
        args.batch_size1 = 128
    elif args.model_name == 'deposit':
        args.epochs1 = 100
        args.epochs2 = 0
        args.batch_size1 = 32
    if 'R3D' in args.img_backbone_name or 'csn' in args.mg_backbone_name\
        or 'R3D' in args.ctx_backbone_name or 'csn' in args.ctx_backbone_name:
        args.img_norm_mode = 'kinetics'
    if args.img_norm_mode in ('kinetics', '0.5', 'activitynet'):
        args.color_order = 'RGB'
    else:
        args.color_order = 'BGR'
    if args.uncertainty != 'gaussian':
        args.logsig_loss_eff = 0

    args.overlap = [args.overlap1, args.overlap2]
    args.epochs = [args.epochs1, args.epochs2]
    args.batch_size = [args.batch_size1, args.batch_size2]
    args.lr = [args.lr1, args.lr2]
    args.backbone_lr = [args.backbone_lr1, args.backbone_lr2]
    args.scheduler_type = [args.scheduler_type1, args.scheduler_type2]

    args.mse_eff = [args.mse_eff1, args.mse_eff2]
    args.pose_mse_eff = [args.pose_mse_eff1, args.pose_mse_eff1]
    args.cls_eff = [args.cls_eff1, args.cls_eff2]

    args.m_settings = {}
    for m in args.modalities:
        args.m_settings[m] = {
            'backbone_name': getattr(args, f'{m}_backbone_name'),
        }
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
    small_set = args.small_set
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
    scheduler_type1 = args.scheduler_type1
    scheduler_type2 = args.scheduler_type2
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
    cls_loss_func = args.cls_loss_func
    cls_eff1 = args.cls_eff1
    cls_eff2 = args.cls_eff2
    logsig_thresh = args.logsig_thresh
    logsig_loss_eff = args.logsig_loss_eff
    logsig_loss_func = args.logsig_loss_func
    mono_sem_eff = args.mono_sem_eff
    stoch_mse_type = args.stoch_mse_type
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

    m_settings = {}

    # conditioned config
    if model_name in ('sgnet', 'sgnet_cvae', 'deposit'):
        cls_eff1 = 0
        cls_eff2 = 0
        mse_eff1 = 1
        traj_format = '0-1ltrb'
    if model_name == 'PCPA':
        mse_eff1 = 0
        mse_eff2 = 0
        if 'JAAD' in test_dataset_names2 or 'JAAD' in test_dataset_names1:
            modalities = ['sklt','ctx', 'traj']
        else:
            modalities = ['sklt','ctx', 'traj', 'ego']
        if '0-1' in sklt_format:
            sklt_format = '0-1coord'
        else:
            sklt_format = 'coord'
        ctx_format = 'ori_local'
    elif model_name == 'ped_graph':
        mse_eff1 = 0
        mse_eff2 = 0
        if 'JAAD' in test_dataset_names2 or 'JAAD' in test_dataset_names1:
            modalities = ['sklt','ctx']
        else:
            modalities = ['sklt','ctx', 'ego']
        if '0-1' in sklt_format:
            sklt_format = '0-1coord'
        else:
            sklt_format = 'coord'
        ctx_format = 'ped_graph'
    elif model_name == 'next':
        cls_eff1 = 1
        mse_eff1 = 1
        epochs1 = 100
        epochs2 = 0
        batch_size1 = 64
        modalities = ['img', 'sklt', 'ctx', 'social', 'traj']
        sklt_format = '0-1coord'
        ctx_format = 'ped_graph'
    elif model_name in ('sgnet', 'sgnet_cvae'):
        epochs1 = 50
        epochs2 = 0
        batch_size1 = 128
    elif model_name == 'deposit':
        epochs1 = 100
        epochs2 = 0
        batch_size1 = 32
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
    makedir(exp_root)
    exp_id = time.strftime("%d%b%Y-%Hh%Mm%Ss")
    exp_num = get_exp_num()
    model_type = model_name
    for m in modalities:
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
    log('----------------------------Load data-----------------------------')
    datasets = [
        {
        'train':{k:None for k in dataset_names1},
        'val':{k:None for k in dataset_names1},
        'test':{k:None for k in dataset_names1},
        },
        {
        'train':{k:None for k in dataset_names1},
        'val':{k:None for k in dataset_names1},
        'test':{k:None for k in dataset_names1},
        },
    ]
    for stage in range(len(datasets)):
        for subset in datasets[stage]:
            _subset = subset
            _overlap = overlap1 if stage == 0 else overlap2
            _small_set = test_small_set
            if subset == 'pre':
                _subset = 'train'
                _small_set = small_set
            elif subset == 'train':
                _small_set = small_set
            for name in datasets[stage][subset]:
                if name == 'TITAN':
                    cur_set = TITAN_dataset(sub_set='default_'+_subset, 
                                            norm_traj=False,
                                            img_norm_mode=img_norm_mode, 
                                            color_order=color_order,
                                            obs_len=obs_len, 
                                            pred_len=pred_len, 
                                            overlap_ratio=_overlap, 
                                            obs_fps=obs_fps,
                                            recog_act=False,
                                            multi_label_cross=False, 
                                            act_sets=act_sets,
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
                    cur_set = PIEDataset(dataset_name=name, 
                                         seq_type='crossing',
                                        subset=_subset,
                                        obs_len=obs_len, 
                                        pred_len=pred_len, 
                                        overlap_ratio=_overlap, 
                                        obs_fps=obs_fps,
                                        do_balance=False, 
                                        bbox_size=(224, 224), 
                                        img_norm_mode=img_norm_mode, 
                                        color_order=color_order,
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
                                        obs_len=obs_len, 
                                        pred_len=pred_len, 
                                        overlap_ratio=_overlap, 
                                        obs_fps=obs_fps,
                                        small_set=_small_set,
                                        augment_mode=augment_mode,
                                        resize_mode=resize_mode,
                                        color_order=color_order, 
                                        img_norm_mode=img_norm_mode,
                                        modalities=modalities,
                                        img_format=img_format,
                                        sklt_format=sklt_format,
                                        ctx_format=ctx_format,
                                        traj_format=traj_format,
                                        ego_format=ego_format
                                        )
                if name == 'bdd100k':
                    cur_set = BDD100kDataset(subsets=_subset,
                                            obs_len=obs_len, 
                                            pred_len=pred_len, 
                                            overlap_ratio=_overlap, 
                                            obs_fps=obs_fps,
                                            color_order=color_order, 
                                            img_norm_mode=img_norm_mode,
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
    for stage in range(len(datasets)):
        for _sub in datasets[stage]:
            for nm in datasets[stage][_sub]:
                if datasets[_sub][nm] is not None:
                    log(f'{_sub} {nm} {len(datasets[_sub][nm])}')
    
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
        _batch_size = batch_size1 if stage == 1 else batch_size2
        train_loaders.append(
            torch.utils.data.DataLoader(concat_train_sets[stage],
                                        batch_size=_batch_size, 
                                        shuffle=shuffle,
                                        num_workers=dataloader_workers,
                                        pin_memory=True)
        )
        val_loaders.append(
            [torch.utils.data.DataLoader(val_sets[stage][i], 
                                        batch_size=_batch_size, 
                                        shuffle=shuffle,
                                        num_workers=dataloader_workers,
                                        pin_memory=True
                                        ) for i in range(len(val_sets[stage]))]
        )
        test_loaders.append(
            [torch.utils.data.DataLoader(test_sets[stage][i], 
                                        batch_size=_batch_size, 
                                        shuffle=shuffle,
                                        num_workers=dataloader_workers,
                                        pin_memory=True
                                        ) for i in range(len(test_sets[stage]))]
        )

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
        sgnet_args.enc_steps = obs_len
        sgnet_args.dec_steps = pred_len
        model = SGNet(sgnet_args)
    elif model_name == 'sgnet_cvae':
        sgnet_args = parse_sgnet_args()
        model = SGNet_CVAE(sgnet_args)
    elif model_name == 'next':

        model = Next(obs_len=obs_len,
                     pred_len=pred_len,
                     action_sets=act_sets,
                     )
    elif model_name == 'deposit':

        raise NotImplementedError()
    else:
        raise NotImplementedError
    model = model.float().to(device)
    model_parallel = torch.nn.parallel.DataParallel(model)
    log('----------------------------Construct optimizer----------------------------')
    # optimizer
    lr_scheduler1 = None
    if 'sgnet' in model_name:
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
    elif model_name == 'next':
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
    elif model_name == 'deposit':
        optimizer1 = torch.optim.Adam(model.parameters(), 
                                      lr=1.0e-3, 
                                      weight_decay=1e-6)
        p1 = int(0.75 * epochs1)
        p2 = int(0.9 * epochs1)
        lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(
            optimizer1, milestones=[p1, p2], gamma=0.1
        )
    else:
        backbone_params, other_params = model.get_pretrain_params()
        opt_specs1 = [{'params': backbone_params, 'lr': backbone_lr1},
                        {'params': other_params, 'lr':lr1}]
        opt_specs2 = [{'params': backbone_params, 'lr': backbone_lr2},
                        {'params': other_params, 'lr':lr2}]
        
        if optim == 'sgd':
            optimizer1 = torch.optim.SGD(opt_specs1, weight_decay=wd)
            optimizer2 = torch.optim.SGD(opt_specs2, weight_decay=wd)
        elif optim == 'adam':
            optimizer1 = torch.optim.Adam(opt_specs1, weight_decay=wd, eps=1e-5)
            optimizer2 = torch.optim.Adam(opt_specs2, weight_decay=wd, eps=1e-5)
        elif optim == 'adamw':
            optimizer1 = torch.optim.AdamW(opt_specs1, weight_decay=wd, eps=1e-5)
            optimizer2 = torch.optim.AdamW(opt_specs2, weight_decay=wd, eps=1e-5)
        else:
            raise NotImplementedError(optim)
        # learning rate scheduler
        if scheduler_type1 == 'step':
            lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer1, 
                                                            step_size=lr_step_size, 
                                                            gamma=lr_step_gamma)
        elif scheduler_type1 == 'cosine':
            lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer1, 
                                                                        T_max=t_max, 
                                                                        eta_min=0)
        elif scheduler_type1 == 'onecycle':
            lr_scheduler1 = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer1, 
                                                                max_lr=lr1*onecycle_div_f, # ?
                                                                epochs=epochs1,
                                                                steps_per_epoch=len(train_loaders[0]),
                                                                div_factor=onecycle_div_f,
                                                                )
        else:
            raise NotImplementedError(scheduler_type1)
        p_warmer1 = warmup.LinearWarmup(optimizer1, 
                                        warmup_period=warm_step1)
        if scheduler_type2 == 'step':
            lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer2, 
                                                        step_size=lr_step_size, 
                                                        gamma=lr_step_gamma)
        elif scheduler_type2 == 'cosine':
            lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer2, 
                                                                    T_max=t_max, 
                                                                    eta_min=0)
        elif scheduler_type2 == 'onecycle':
            lr_scheduler2 = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer2, 
                                                                max_lr=lr2*onecycle_div_f,
                                                                epochs=epochs2,
                                                                steps_per_epoch=len(train_loaders[1]),
                                                                div_factor=onecycle_div_f,
                                                                )
        else:
            raise NotImplementedError(scheduler_type2)

    # init curve dict
    metric_dict = {
            'cls': {},
            'traj_mse':[],
            'pose_mse':[],
            'contrast_loss':[],
            'logsig_loss':[],
            'mono_sem_loss': [],
        }
    for k in act_sets:
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
    best_val_res = {
        'cls': {},
        'traj_mse':[],
        'pose_mse':[],
    }
    for k in act_sets:
        best_val_res['cls'][k] = {'acc':[],
                                'auc':[],
                                'f1':[],
                                'map':[],}
    best_test_res = copy.deepcopy(best_val_res)
    # loss params
    loss_params = [{
        'mse_eff': mse_eff1,
        'pose_mse_eff': pose_mse_eff1,
        'stoch_mse_type': stoch_mse_type,
        'n_sampling': n_pred_sampling,
        'cls_eff':cls_eff1,
        'cls_loss': cls_loss_func,
        'logsig_loss_func': logsig_loss_func,
        'logsig_loss_eff':logsig_loss_eff,
        'logsig_thresh':logsig_thresh,
        'mono_sem_eff': mono_sem_eff,
    },
    {
        'mse_eff': mse_eff2,
        'pose_mse_eff': pose_mse_eff2,
        'stoch_mse_type': stoch_mse_type,
        'n_sampling': n_pred_sampling,
        'cls_eff':cls_eff2,
        'cls_loss': cls_loss_func,
        'logsig_loss_func': logsig_loss_func,
        'logsig_loss_eff':logsig_loss_eff,
        'logsig_thresh':logsig_thresh,
        'mono_sem_eff': mono_sem_eff,
    }]

    # stage 1
    log('----------------------------STAGE 1----------------------------')
    for e in range(1, 1+epochs1):
        log(f' stage 1 epoch {e}')
        log('Train')
        train_res = train_test_epoch(model=model_parallel,
                                    model_name=model_name,
                                    dataloader=train_loaders[0],
                                    optimizer=optimizer1,
                                    scheduler=lr_scheduler1,
                                    log=log,
                                    device=device,
                                    modalities=modalities,
                                    loss_params=loss_params[0],
                                    )
        # add results to curve
        if 'cls' in train_res:
            for act_set in train_res['cls']:
                for metric in train_res['cls'][act_set]:
                    curve_dict['concat']['train']['cls'][act_set][metric]\
                        .append(train_res['cls'][act_set][metric])
        for metric in train_res:
            if metric == 'cls':
                continue
            curve_dict['concat']['train'][metric].append(train_res[metric])
        # validation and test
        if e%test_every == 0:
            log('Val')
            for val_loader in val_loaders[0]:
                cur_dataset = val_loader.dataset.dataset_name
                log(cur_dataset)
                val_res = train_test_epoch(model=model_parallel,
                                            model_name=model_name,
                                            dataloader=val_loader,
                                            optimizer=None,
                                            scheduler=lr_scheduler1,
                                            log=log,
                                            device=device,
                                            modalities=modalities,
                                            loss_params=loss_params[0],
                                            )
                if 'cls' in val_res:
                    for act_set in val_res['cls']:
                        for metric in val_res['cls'][act_set]:
                            curve_dict[cur_dataset]['val']['cls'][act_set][metric]\
                                .append(val_res['cls'][act_set][metric])
                for metric in val_res:
                    if metric == 'cls':
                        continue
                    curve_dict[cur_dataset]['val'][metric].append(val_res[metric])
            log('Test')
            for test_loader in test_loaders[0]:
                cur_dataset = test_loader.dataset.dataset_name
                log(cur_dataset)
                test_res = train_test_epoch(model=model_parallel,
                                            model_name=model_name,
                                            dataloader=test_loader,
                                            optimizer=None,
                                            scheduler=lr_scheduler1,
                                            log=log,
                                            device=device,
                                            modalities=modalities,
                                            loss_params=loss_params[0],
                                            )
                if 'cls' in test_res:
                    for act_set in test_res['cls']:
                        for metric in test_res['cls'][act_set]:
                            curve_dict[cur_dataset]['test']['cls'][act_set][metric]\
                                .append(test_res['cls'][act_set][metric])
                            curve_list = [
                                curve_dict[cur_dataset]['val']['cls'][act_set][metric],
                                curve_dict[cur_dataset]['test']['cls'][act_set][metric],
                                curve_dict['concat']['train']['cls'][act_set][metric],
                            ]
                            draw_curves2(path=os.path.join(train_test_plot_dir, 
                                                   cur_dataset+'_'+act_set+'_'+metric+'.png'), 
                                        val_lists=curve_list,
                                        labels=['val', 'test', 'train'],
                                        colors=['g', 'b', 'r'],
                                        vis_every=vis_every)
                for metric in test_res:
                    if metric == 'cls':
                        continue
                    curve_dict[cur_dataset]['test'][metric].append(test_res[metric])
                    curve_list = [
                        curve_dict[cur_dataset]['val'][metric],
                        curve_dict[cur_dataset]['test'][metric],
                        curve_dict['concat']['train'][metric]
                    ]
                    draw_curves2(path=os.path.join(train_test_plot_dir, 
                                                   cur_dataset+'_'+metric+'.png'), 
                                val_lists=curve_list,
                                labels=['val', 'test', 'train'],
                                colors=['g', 'b', 'r'],
                                vis_every=vis_every)
            # save best results
            if key_metric in ('traj_mse', 'pose_mse'):
                cur_key_res = sum([curve_dict[d]['val'][key_metric][-1] for d in test_dataset_names1]) \
                    / len(test_dataset_names1)
                if cur_key_res < best_val_res[key_metric]:
                    update_best_res(best_val_res, best_test_res, curve_dict, test_dataset_names1)
            elif key_metric in ('f1', 'acc', 'auc', 'map'):
                cur_key_res = sum([curve_dict[d]['val']['cls'][key_act_set][key_metric][-1] for d in test_dataset_names1]) \
                    / len(test_dataset_names1)
                if cur_key_res > best_val_res['cls'][key_act_set][key_metric]:
                    update_best_res(best_val_res, best_test_res, curve_dict, test_dataset_names1)
            if local_rank == 0 or not ddp:
                save_model(model=model, model_dir=ckpt_dir, 
                            model_name=str(e) + '_',
                            log=log)
            log(f'current best val results: {best_val_res}')
            log(f'current best test results: {best_test_res}')
    log('----------------------------STAGE 2----------------------------')
    for e in range(1, 1+epochs2):
        log(f' stage 2 epoch {e}')
        train_test_epoch(model=model_parallel,
                         model_name=model_name,
                         dataloader=train_loaders[1],
                         optimizer=optimizer2,
                         scheduler=lr_scheduler2,
                         log=log,
                         device=device,
                         modalities=modalities,
                         loss_params=loss_params[1],
                         )
    # add results to curve
        if 'cls' in train_res:
            for act_set in train_res['cls']:
                for metric in train_res['cls'][act_set]:
                    curve_dict['concat']['train']['cls'][act_set][metric]\
                        .append(train_res['cls'][act_set][metric])
        for metric in train_res:
            if metric == 'cls':
                continue
            curve_dict['concat']['train'][metric].append(train_res[metric])
        # validation and test
        if e%test_every == 0:
            log('Val')
            for val_loader in val_loaders[1]:
                cur_dataset = val_loader.dataset.dataset_name
                log(cur_dataset)
                val_res = train_test_epoch(model=model_parallel,
                                            model_name=model_name,
                                            dataloader=val_loader,
                                            optimizer=None,
                                            scheduler=lr_scheduler2,
                                            log=log,
                                            device=device,
                                            modalities=modalities,
                                            loss_params=loss_params[1],
                                            )
                if 'cls' in val_res:
                    for act_set in val_res['cls']:
                        for metric in val_res['cls'][act_set]:
                            curve_dict[cur_dataset]['val']['cls'][act_set][metric]\
                                .append(val_res['cls'][act_set][metric])
                for metric in val_res:
                    if metric == 'cls':
                        continue
                    curve_dict[cur_dataset]['val'][metric].append(val_res[metric])
            log('Test')
            for test_loader in test_loaders[1]:
                cur_dataset = test_loader.dataset.dataset_name
                log(cur_dataset)
                test_res = train_test_epoch(model=model_parallel,
                                            model_name=model_name,
                                            dataloader=test_loader,
                                            optimizer=None,
                                            scheduler=lr_scheduler2,
                                            log=log,
                                            device=device,
                                            modalities=modalities,
                                            loss_params=loss_params[1],
                                            )
                if 'cls' in test_res:
                    for act_set in test_res['cls']:
                        for metric in test_res['cls'][act_set]:
                            curve_dict[cur_dataset]['test']['cls'][act_set][metric]\
                                .append(test_res['cls'][act_set][metric])
                            curve_list = [
                                curve_dict[cur_dataset]['val']['cls'][act_set][metric],
                                curve_dict[cur_dataset]['test']['cls'][act_set][metric],
                                curve_dict['concat']['train']['cls'][act_set][metric],
                            ]
                            draw_curves2(path=os.path.join(train_test_plot_dir, 
                                                   cur_dataset+'_'+act_set+'_'+metric+'.png'), 
                                        val_lists=curve_list,
                                        labels=['val', 'test', 'train'],
                                        colors=['g', 'b', 'r'],
                                        vis_every=vis_every)
                for metric in test_res:
                    if metric == 'cls':
                        continue
                    curve_dict[cur_dataset]['test'][metric].append(test_res[metric])
                    curve_list = [
                        curve_dict[cur_dataset]['val'][metric],
                        curve_dict[cur_dataset]['test'][metric],
                        curve_dict['concat']['train'][metric]
                    ]
                    draw_curves2(path=os.path.join(train_test_plot_dir, 
                                                   cur_dataset+'_'+metric+'.png'), 
                                val_lists=curve_list,
                                labels=['val', 'test', 'train'],
                                colors=['g', 'b', 'r'],
                                vis_every=vis_every)
            # save best results
            if key_metric in ('traj_mse', 'pose_mse'):
                cur_key_res = sum([curve_dict[d]['val'][key_metric][-1] for d in test_dataset_names1]) \
                    / len(test_dataset_names1)
                if cur_key_res < best_val_res[key_metric]:
                    update_best_res(best_val_res, 
                                    best_test_res, 
                                    curve_dict, 
                                    test_dataset_names2)
            elif key_metric in ('f1', 'acc', 'auc', 'map'):
                cur_key_res = sum([curve_dict[d]['val']['cls'][key_act_set][key_metric][-1] for d in test_dataset_names1]) \
                    / len(test_dataset_names1)
                if cur_key_res > best_val_res['cls'][key_act_set][key_metric]:
                    update_best_res(best_val_res, 
                                    best_test_res, 
                                    curve_dict, 
                                    test_dataset_names2)
            if local_rank == 0 or not ddp:
                save_model(model=model, model_dir=ckpt_dir, 
                            model_name=str(e) + '_',
                            log=log)
            log(f'current best val results: {best_val_res}')
            log(f'current best test results: {best_test_res}')
    logclose()
    with open(os.path.join(train_test_plot_dir, 'curve_dict.pkl'), 'wb') as f:
        pickle.dump(curve_dict, f)




def update_best_res(best_val_res,
                    best_test_res,
                    curve_dict,
                    test_dataset_names,
                    ):
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


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    args = get_args()
    if world_size > 1 and args.ddp:
        mp.spawn(main, args=(args),  nprocs=world_size)
    else:
        main(rank=0, world_size=world_size, args=args)