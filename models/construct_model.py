from models.PCPA import PCPA
from models.ped_graph23 import PedGraph
from models.SGNet import SGNet, parse_sgnet_args, parse_sgnet_args2
from models.SGNet_CVAE import SGNet_CVAE
from models.next import Next
from models.deposit import Deposit, deposit_config
from models.PedSpace import PedSpace
from models.SLENN import build_sle, img_setting, sk_setting, ctx_setting, traj_setting, ego_setting


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
    elif args.model_name == 'SLENN':
        model = build_sle(args, 
                          img_setting,
                          sk_setting,
                          ctx_setting,
                          traj_setting,
                          ego_setting)
    elif args.model_name == 'pedspace':
        model = PedSpace(args=args,
                         device=device,
                         )
    return model