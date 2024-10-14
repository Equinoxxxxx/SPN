import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='main args')
    # device
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--ddp", default=False, type=bool)
    # data
    parser.add_argument('--dataset_names1', type=str, default='TITAN_PIE_nuscenes_bdd100k')
    parser.add_argument('--test_dataset_names1', type=str, default='TITAN_PIE_nuscenes_bdd100k')
    parser.add_argument('--dataset_names2', type=str, default='TITAN_PIE')
    parser.add_argument('--test_dataset_names2', type=str, default='TITAN_PIE')
    parser.add_argument('--small_set', type=float, default=0)
    parser.add_argument('--test_small_set', type=float, default=0)
    parser.add_argument('--obs_len', type=int, default=4)
    parser.add_argument('--pred_len', type=int, default=4)
    parser.add_argument('--obs_fps', type=int, default=2)
    parser.add_argument('--apply_tte', type=int, default=1)
    parser.add_argument('--test_apply_tte', type=int, default=1)
    parser.add_argument('--augment_mode', type=str, default='none')
    parser.add_argument('--img_norm_mode', type=str, default='torch')
    parser.add_argument('--model_color_order', type=str, default='BGR')
    parser.add_argument('--resize_mode', type=str, default='even_padded')
    parser.add_argument('--overlap1', type=float, default=0.5)
    parser.add_argument('--overlap2', type=float, default=0.5)
    parser.add_argument('--test_overlap', type=float, default=0.5)
    parser.add_argument('--max_n_neighbor', type=int, default=10)
    parser.add_argument('--dataloader_workers', type=int, default=8)
    parser.add_argument('--shuffle', type=int, default=1)
    # train
    parser.add_argument('--reg_first', type=int, default=1)
    parser.add_argument('--epochs1', type=int, default=30)
    parser.add_argument('--epochs2', type=int, default=30)
    parser.add_argument('--warm_step1', type=int, default=1)
    parser.add_argument('--batch_size1', type=int, default=64)
    parser.add_argument('--batch_size2', type=int, default=64)
    parser.add_argument('--test_every', type=int, default=2)
    parser.add_argument('--explain_every', type=int, default=10)
    parser.add_argument('--vis_every', type=int, default=2)
    parser.add_argument('--lr1', type=float, default=0.001)
    parser.add_argument('--lr2', type=float, default=0.001)
    parser.add_argument('--backbone_lr1', type=float, default=0.0001)
    parser.add_argument('--backbone_lr2', type=float, default=0.0001)
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
    parser.add_argument('--mse_eff1', type=float, default=1.)
    parser.add_argument('--mse_eff2', type=float, default=0.1)
    parser.add_argument('--pose_mse_eff1', type=float, default=1.)
    parser.add_argument('--pose_mse_eff2', type=float, default=0.1)
    parser.add_argument('--cls_loss_func', type=str, default='weighted_ce')
    parser.add_argument('--cls_eff1', type=float, default=0.)
    parser.add_argument('--cls_eff2', type=float, default=1.)
    parser.add_argument('--logsig_thresh', type=float, default=100)
    parser.add_argument('--logsig_loss_eff', type=float, default=0.1)
    parser.add_argument('--logsig_loss_func', type=str, default='kl')
    parser.add_argument('--diversity_loss_func', type=str, default='triangular')
    parser.add_argument('--diversity_loss_eff', type=float, default=0)
    parser.add_argument('--mono_sem_eff', type=float, default=0.01)
    parser.add_argument('--mono_sem_l1_eff', type=float, default=0.01)
    parser.add_argument('--mono_sem_align_func', type=str, default='cosine_simi')
    parser.add_argument('--mono_sem_align_eff', type=float, default=0.01)
    parser.add_argument('--cluster_loss_eff', type=float, default=0.01)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--topk_metric', type=str, default='activation')
    parser.add_argument('--stoch_mse_type', type=str, default='best')
    # model
    parser.add_argument('--pretrain_mode', type=str, default='contrast')
    parser.add_argument('--model_name', type=str, default='pedspace')
    parser.add_argument('--pair_mode', type=str, default='pair_wise')
    parser.add_argument('--simi_func', type=str, default='dot_prod')
    parser.add_argument('--bridge_m', type=str, default='sk')
    parser.add_argument('--n_proto', type=int, default=50)
    parser.add_argument('--linear_proto_enc', type=int, default=1)
    parser.add_argument('--proj_dim', type=int, default=512)
    parser.add_argument('--pool', type=str, default='max')
    parser.add_argument('--n_proj_layer', type=int, default=1)
    parser.add_argument('--proj_bias', type=int, default=1)
    parser.add_argument('--proj_norm', type=str, default='ln')
    parser.add_argument('--proj_actv', type=str, default='leakyrelu')
    parser.add_argument('--mm_fusion_mode', type=str, default='no_fusion', help='avg/gaussian/no_fusion')
    parser.add_argument('--uncertainty', type=str, default='none')
    parser.add_argument('--n_pred_sampling', type=int, default=5)
    # explain
    parser.add_argument('--topk_metric_explain', type=str, default='activation')
    parser.add_argument('--topk_explain', type=int, default=6)
    parser.add_argument('--head_fusion', type=str, default='mean')
    parser.add_argument('--test_customized_proto', type=int, default=1)
    parser.add_argument('--proto_rank_criteria', type=str, default='num_select')
    parser.add_argument('--proto_value_to_rank', type=str, default='abs_weight')
    parser.add_argument('--proto_num_select', type=int, default=10)
    # modality
    parser.add_argument('--modalities', type=str, default='sklt_ctx_traj_ego_social')
    # img settingf
    parser.add_argument('--img_format', type=str, default='')
    parser.add_argument('--img_backbone_name', type=str, default='deeplabv3_resnet50')
    # sk setting
    parser.add_argument('--sklt_format', type=str, default='0-1coord')
    parser.add_argument('--sklt_backbone_name', type=str, default='transformerencoder1D')
    # ctx setting
    parser.add_argument('--ctx_format', type=str, default='ori_local')
    parser.add_argument('--seg_cls', type=str, default='person,vehicles,roads,traffic_lights')
    parser.add_argument('--ctx_backbone_name', type=str, default='deeplabv3_resnet50')
    # social setting
    parser.add_argument('--social_backbone_name', type=str, default='transformerencoder1D')
    parser.add_argument('--social_format', type=str, default='rel_loc')
    # traj setting
    parser.add_argument('--traj_format', type=str, default='0-1ltrb')
    parser.add_argument('--traj_backbone_name', type=str, default='transformerencoder1D')
    # ego setting
    parser.add_argument('--ego_format', type=str, default='accel')
    parser.add_argument('--ego_backbone_name', type=str, default='transformerencoder1D')
    # decoder setting
    parser.add_argument('--traj_dec_name', type=str, default='deposit')
    parser.add_argument('--pose_dec_name', type=str, default='deposit')
    args = parser.parse_args()

    return args


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
    if args.model_name != 'pedspace':
        args.mono_sem_eff = 0
        args.mono_sem_l1_eff = 0
        args.mono_sem_align_eff = 0
        args.logsig_loss_eff = 0
    if args.model_name in ('sgnet', 'sgnet_cvae', 'deposit'):
        args.cls_eff1 = 0
        args.epochs2 = 0
        args.traj_format = '0-1ltrb'
    if args.model_name == 'PCPA':
        args.batch_size1 = 8
        args.cls_eff1 = 1
        args.mse_eff1 = 0
        args.pose_mse_eff1 = 0
        args.epochs2 = 0
        args.mse_eff2 = 0
        if 'JAAD' in args.test_dataset_names2 or 'JAAD' in args.test_dataset_names1:
            args.modalities = ['sklt','ctx', 'traj']
        else:
            args.modalities = ['sklt','ctx', 'traj', 'ego']
        if '0-1' in args.sklt_format:
            args.sklt_format = '0-1coord'
        else:
            args.sklt_format = 'coord'
        args.ctx_format = 'ori_local'
        args.ctx_backbone_name = 'C3D_t4'
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
        # args.cls_eff1 = 1
        # args.mse_eff1 = 1
        # args.epochs1 = 100
        # args.epochs2 = 0
        args.pose_mse_eff1 = 0
        args.pose_mse_eff2 = 0
        args.batch_size1 = 64
        args.modalities = ['img', 'sklt', 'ctx', 'social', 'traj']
        args.sklt_format = '0-1coord'
        args.ctx_format = 'ped_graph'
    elif args.model_name in ('sgnet', 'sgnet_cvae'):
        args.epochs1 = 50
        args.epochs2 = 0
        args.cls_eff1 = 0
        args.mse_eff1 = 1
        args.pose_mse_eff1 = 0
        args.batch_size1 = 128
        args.key_metric = 'traj_mse'
    elif args.model_name == 'deposit':
        args.epochs1 = 100
        args.epochs2 = 0
        args.cls_eff1 = 0
        args.mse_eff1 = 0
        args.pose_mse_eff1 = 1
        args.batch_size1 = 32
        args.key_metric = 'pose_mse'
    if 'R3D' in args.img_backbone_name or 'csn' in args.img_backbone_name\
        or 'R3D' in args.ctx_backbone_name or 'csn' in args.ctx_backbone_name:
        args.img_norm_mode = 'kinetics'
    
    if args.img_norm_mode in ('kinetics', '0.5', 'activitynet'):
        args.model_color_order = 'RGB'
    if args.mm_fusion_mode != 'gaussian':
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
    if args.act_sets != ['cross']:
        args.dataset_names = [['TITAN'], ['TITAN']]
        args.test_dataset_names = [['TITAN'], ['TITAN']]
    for i in range(len(args.dataset_names)):
        if 'nuscenes' in args.dataset_names[i] or 'bdd100k' in args.dataset_names[i]:
            args.cls_eff[i] = 0
    if len(args.act_sets) == 1:
        args.key_act_set = args.act_sets[0]
    args.m_settings = {}
    for m in args.modalities:
        args.m_settings[m] = {
            'backbone_name': getattr(args, f'{m}_backbone_name'),
        }
    return args