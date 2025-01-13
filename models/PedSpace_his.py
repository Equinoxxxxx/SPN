import torch
import torch.nn as nn
import numpy as np
from .backbones import create_backbone, FLATTEN_DIM, LAST_DIM
from tools.datasets.TITAN import ACT_SET_TO_N_CLS
from .deposit import Deposit, deposit_config


ACTIVATION_FUNC = {
    'relu': nn.ReLU,
    'leakyrelu': nn.LeakyReLU,
}


def update_gaussian(mu1, mu2, logsig1, logsig2):
    _eps = 1e-5
    sig1, sig2 = torch.exp(logsig1), torch.exp(logsig2)
    eff1 = sig2 / (sig1 + sig2 + _eps)
    eff2 = sig1 / (sig1 + sig2 + _eps)
    mu = eff1 * mu1 + eff2 * mu2
    sig = sig1 * sig2 / (sig1 + sig2 + _eps)
    logsig = torch.log(sig + _eps)
    return mu, logsig, (eff1, eff2)


class PedSpace(nn.Module):
    def __init__(self,
                 args,
                 device,
                 ) -> None:
        super().__init__()
        self.device = device
        self.modalities = args.modalities
        self.act_sets = args.act_sets
        self.n_proto = args.n_proto
        self.proj_dim = args.proj_dim
        self.linear_proto_enc = args.linear_proto_enc
        self.mm_fusion_mode = args.mm_fusion_mode
        self.n_pred_sampling = args.n_pred_sampling
        self.traj_dec_name = args.traj_dec_name
        self.pose_dec_name = args.pose_dec_name
        self.do_pred_traj = False
        self.do_pred_pose = False
        if args.mse_eff1 > 0 or args.mse_eff2 > 0:
            self.do_pred_traj = True
        if args.pose_mse_eff1 > 0 or args.pose_mse_eff2 > 0:
            self.do_pred_pose = True
        self.logit_scale = nn.parameter.Parameter(
            torch.ones([]) * np.log(1 / 0.07))
        self.all_sparsity = None
        # encoders
        self.encoders = {}
        for m in self.modalities:
            self.encoders[m] = SingleBranch(args, m)
        self.encoders = nn.ModuleDict(self.encoders)
        # fusion
        if self.mm_fusion_mode == 'gaussian':
            self.mu_enc = {}
            self.sig_enc = {}
            for m in self.modalities:
                self.mu_enc[m] = nn.Sequential(
                    nn.Linear(self.proj_dim, self.proj_dim),
                    nn.ReLU(),
                    nn.Linear(self.proj_dim, self.proj_dim)
                    )
                self.sig_enc[m] = nn.Sequential(
                    nn.Linear(self.proj_dim, self.proj_dim),
                    nn.ReLU(),
                    nn.Linear(self.proj_dim, self.proj_dim)
                    )
            self.mu_enc = nn.ModuleDict(self.mu_enc)
            self.sig_enc = nn.ModuleDict(self.sig_enc)
        # projection to proto
        self.proto_enc = nn.Linear(in_features=self.proj_dim, 
                                   out_features=self.n_proto, 
                                   bias=False)
        self.proto_enc_actv = nn.ReLU()
        # decoders
        self.proto_dec = {}
        for act_set in self.act_sets:
            self.proto_dec[act_set] = nn.Linear(
                self.n_proto,
                ACT_SET_TO_N_CLS[act_set], 
                bias=False)
        self.proto_dec = nn.ModuleDict(self.proto_dec)
        if self.do_pred_traj:
            if self.traj_dec_name == 'deposit':
                self.traj_decoder = Deposit(deposit_config, 
                                            device, 
                                            target_dim=4, 
                                            w_mm_cond=1,
                                            mm_cond_dim=self.proj_dim, 
                                            modality='traj')
            else:
                raise ValueError(self.traj_dec_name)
        if self.do_pred_pose:
            if self.pose_dec_name == 'deposit':
                self.pose_decoder = Deposit(deposit_config, 
                                            device, 
                                            target_dim=17*2,
                                            w_mm_cond=1,
                                            mm_cond_dim=self.proj_dim,
                                            modality='sklt')
            else:
                raise ValueError(self.pose_dec_name)
    
    def forward(self, batch, is_train=1):
        # import pdb; pdb.set_trace()
        inputs, targets = batch
        feat = {}
        out = {}
        for m in self.modalities:
            cur_feat, cur_out = self.encoders[m](inputs[m])
            feat[m] = cur_feat
            out[m] = cur_out
            
        # fusion
        modality_effs = {}
        if self.mm_fusion_mode == 'gaussian':
            mu, sig = None, None
            for m in self.modalities:
                if mu is None:
                    try:
                        mu = self.mu_enc[m](out[m])
                    except:
                        import pdb; pdb.set_trace()
                    sig = self.sig_enc[m](out[m])
                    modality_effs[m] = None
                else:
                    mu, sig, effs = update_gaussian(mu, self.mu_enc[m](out[m]), sig, self.sig_enc[m](out[m]))
                    if len(modality_effs.keys()) == 1:
                        modality_effs[list(modality_effs.keys())[0]] = effs[0].mean(-1)
                    modality_effs[m] = effs[1].mean(-1)
            eps = torch.randn(mu.shape[0], mu.shape[1], device=mu.device)  # B d
            feat_fused = mu + eps*torch.exp(sig)  # B d
        
        elif self.mm_fusion_mode in ('avg', 'no_fusion'):
            modality_effs = {m: torch.ones(inputs[m].shape[0]).to(out[m].device) for m in self.modalities}
            feat_fused = torch.stack(list(out.values()), dim=2).mean(dim=2)  # B d M -> B d
        else:
            raise ValueError(self.mm_fusion_mode)
        # projection to proto
        proto_simi = self.proto_enc(feat_fused)  # B n_proto
        if not self.linear_proto_enc:
            proto_simi = self.proto_enc_actv(proto_simi)
        # mm proto_simi
        mm_proto_simi = None
        if self.mm_fusion_mode == 'no_fusion':
            mm_proto_simi = {}
            for m in self.modalities:
                mm_proto_simi[m] = self.proto_enc(out[m])
                if not self.linear_proto_enc:
                    mm_proto_simi[m] = self.proto_enc_actv(mm_proto_simi[m])
        # decoders
        preds = {'feat': feat,
                 'enc_out': out,
                 'modality_effs': modality_effs,
                'proto_simi': proto_simi,
                'mm_proto_simi': mm_proto_simi,
                'cls_logits': {}, 
                 'pred_traj': None, 
                 'pred_pose': None,
                 'traj_loss': 0,
                 'pose_loss': 0}
        for act_set in self.act_sets:
            preds['cls_logits'][act_set] = self.proto_dec[act_set](proto_simi)
        # pred traj
        if self.do_pred_traj:
            if self.traj_dec_name == 'deposit':
                deposit_out = self.traj_decoder(batch, 
                                        n_samples=self.n_pred_sampling,
                                        mm_cond=feat_fused,
                                        is_train=is_train)
                preds['traj_loss'] = deposit_out['loss']
                preds['pred_traj'] = deposit_out['pred']
            else:
                raise ValueError(self.traj_dec_name)
        # pred pose
        if self.do_pred_pose:
            if self.pose_dec_name == 'deposit':
                deposit_out = self.pose_decoder(batch, 
                                        n_samples=self.n_pred_sampling,
                                        mm_cond=feat_fused,
                                        is_train=is_train)
                preds['pose_loss'] = deposit_out['loss']
                preds['pred_pose'] = deposit_out['pred']
            else:
                raise ValueError(self.pose_dec_name)
        return preds
    
    def get_backbone_params(self):
        bb_params = []
        other_params = []
        for n, p in self.named_parameters():
            if 'backbone' in n and \
                ('img' in n or 'ctx' in n or ('sklt' in n and 'poseC3D' in n)):
                bb_params.append(p)
            else:
                other_params.append(p)
        
        return bb_params, other_params


class SingleBranch(nn.Module):
    def __init__(self,
                 args,
                 modality) -> None:
        super().__init__()
        self.modality = modality
        self.proj_dim = args.proj_dim
        self.n_proj_layer = args.n_proj_layer
        self.proj_bias = args.proj_bias
        self.proj_norm = args.proj_norm
        self.proj_actv = args.proj_actv
        self.head_fusion = args.head_fusion
        self.backbone_name = getattr(args, f'{self.modality}_backbone_name')
        self.backbone = create_backbone(self.backbone_name, 
                                        modality=self.modality,
                                        args=args)
        self.proj_pooling = None
        # pooling layer
        if 'flat' in self.backbone_name:
            self.proj_pooling = nn.Flatten()
        elif '3D' in self.backbone_name:
            self.proj_pooling = nn.AdaptiveAvgPool3d(1) if args.pool == 'avg' else nn.AdaptiveMaxPool3d(1)
        elif '1D' in self.backbone_name:
            self.proj_pooling = nn.AdaptiveAvgPool1d(1) if args.pool == 'avg' else nn.AdaptiveMaxPool1d(1)
        elif 'deeplab' in self.backbone_name or 'vit' in self.backbone_name or \
            'pedgraphconv' in self.backbone_name or '2D' in self.backbone_name:
            self.proj_pooling = nn.AdaptiveAvgPool2d(1) if args.pool == 'avg' else nn.AdaptiveMaxPool2d(1)
        feat_dim = LAST_DIM[self.backbone_name]
        # proj layer
        self.proj_layers = []
        in_dim = feat_dim
        for i in range(self.n_proj_layer - 1):
            self.proj_layers.append(nn.Linear(in_dim, self.proj_dim, bias=self.proj_bias))
            if self.proj_norm == 'ln':
                self.proj_layers.append(nn.LayerNorm(self.proj_dim))
            elif self.proj_norm == 'bn':
                self.proj_layers.append(nn.BatchNorm1d(self.proj_dim))
            if self.proj_actv is not None:
                self.proj_layers.append(ACTIVATION_FUNC[self.proj_actv]())
            in_dim = self.proj_dim
        self.proj_layers.append(nn.Linear(in_dim, self.proj_dim, bias=self.proj_bias))
        self.proj_layers = nn.Sequential(*self.proj_layers)

    def forward(self, x):
        if self.modality == 'social' and 'transformer' in self.backbone_name:
            # B K T 5 --> B 5 T K
            x = x.permute(0, 3, 2, 1)
        feat = self.backbone(x)
        
        out = feat
        if self.proj_pooling is not None:
            out = self.proj_pooling(feat)
        # B C ...
        B, C = out.size(0), out.size(1)
        out = out.reshape(B, C)  # B C 1 --> B C
        out = self.proj_layers(out)

        if 'transformer' in self.backbone_name:
            try:
                feat = self.backbone.attn_list[-1]  # B nhead T T
            except:
                import pdb; pdb.set_trace()
            feat = feat.sum(2)  # B nhead T
            if self.head_fusion == 'mean':
                feat = feat.mean(1) # B T
            else:
                raise NotImplementedError(self.head_fusion)
        elif 'vit' in self.backbone_name:
            raise NotImplementedError(self.backbone_name)
        return feat, out