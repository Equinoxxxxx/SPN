import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .backbones import create_backbone, FLATTEN_DIM, LAST_DIM
from tools.datasets.TITAN import ACT_SET_TO_N_CLS


class PCPA(nn.Module):
    def __init__(self, 
                 h_dim=256,
                 q_modality='ego',
                 modalities=['sklt','ctx','traj','ego'],
                 ctx_bb_nm='C3D_t4_clean',
                 act_sets=['cross'],
                 proj_norm='bn',
                 proj_actv='silu',
                 pretrain=True,
                 n_mlp=1,
                 proj_dim=256,
                 ) -> None:
        super(PCPA, self).__init__()
        self.model_name = 'PCPA'
        self.h_dim = h_dim
        self.q_modality = q_modality
        self.modalities = modalities
        self.ctx_bb_nm = ctx_bb_nm
        self.act_sets = act_sets
        self.proj_norm = proj_norm
        self.proj_actv = proj_actv
        self.pretrain = pretrain
        self.n_mlp = n_mlp
        self.proj_dim = proj_dim if proj_dim > 0 else self.h_dim

        # init contrast scale factor
        self.logit_scale = nn.parameter.Parameter(
            torch.ones([]) * np.log(1 / 0.07))

        self.encoders = {}
        self.pools = {}
        self.proj = {}
        for k in self.modalities:
            self.proj[k] = []
            for i in range(self.n_mlp):
                if i == 0:
                    self.proj[k].append(nn.Linear(self.h_dim, self.proj_dim))
                else:
                    self.proj[k].append(nn.Linear(self.proj_dim, self.proj_dim))
                if self.proj_norm == 'ln':
                    self.proj[k].append(nn.LayerNorm(self.proj_dim))
                elif self.proj_norm == 'bn':
                    self.proj[k].append(nn.BatchNorm1d(self.proj_dim))
                if self.proj_actv == 'silu':
                    self.proj[k].append(nn.SiLU())
                elif self.proj_actv == 'relu':
                    self.proj[k].append(nn.ReLU())
                elif self.proj_actv == 'leakyrelu':
                    self.proj[k].append(nn.LeakyReLU())
            if self.n_mlp > 1:
                self.proj[k].append(nn.Linear(self.proj_dim, self.proj_dim))
            self.proj[k] = nn.Sequential(*self.proj[k])
        self.proj = nn.ModuleDict(self.proj)
        self.att_w = {}
        self.att_out = {}
        self.dropout = {}
        if 'traj' in self.modalities:
            self.encoders['traj'] = nn.GRU(4, self.h_dim, batch_first=True)
            self.att_w['traj'] = nn.Linear(self.h_dim, self.h_dim, bias=False)
            self.att_out['traj'] = nn.Linear(self.h_dim * 2, self.h_dim, bias=False)
            self.dropout['traj'] = nn.Dropout(0.5)
            self.pools['traj'] = nn.AdaptiveAvgPool1d(1)
        if 'sklt' in self.modalities:
            self.encoders['sklt'] = nn.GRU(34, self.h_dim, batch_first=True)
            self.att_w['sklt'] = nn.Linear(self.h_dim, self.h_dim, bias=False)
            self.att_out['sklt'] = nn.Linear(self.h_dim * 2, self.h_dim, bias=False)
            self.dropout['sklt'] = nn.Dropout(0.5)
            self.pools['sklt'] = nn.AdaptiveAvgPool1d(1)
        if 'ego' in self.modalities:
            self.encoders['ego'] = nn.GRU(1, self.h_dim, batch_first=True)
            self.att_w['ego'] = nn.Linear(self.h_dim, self.h_dim, bias=False)
            self.att_out['ego'] = nn.Linear(self.h_dim * 2, self.h_dim, bias=False)
            self.dropout['ego'] = nn.Dropout(0.5)
            self.pools['ego'] = nn.AdaptiveAvgPool1d(1)
        if 'ctx' in self.modalities:
            self.encoders['ctx'] = create_backbone(backbone_name=self.ctx_bb_nm, last_dim=487)
            self.ctx_embedder = nn.Linear(8192, self.h_dim, bias=False)
            self.ctx_sigm = nn.Sigmoid()
            self.pools['ctx'] = nn.AdaptiveAvgPool3d(1)
        self.encoders = nn.ModuleDict(self.encoders)
        self.att_w = nn.ModuleDict(self.att_w)
        self.att_out = nn.ModuleDict(self.att_out)
        self.dropout = nn.ModuleDict(self.dropout)

        self.modal_att_w = nn.Linear(self.h_dim, self.h_dim, bias=False)
        self.modal_att_out = nn.Linear(self.h_dim * 2, self.h_dim, bias=False)

        # last layers
        self.final_layers = {}
        for act_set in self.act_sets:
            self.final_layers[act_set] = nn.Linear(self.h_dim, ACT_SET_TO_N_CLS[act_set], bias=False)
        self.final_layers = nn.ModuleDict(self.final_layers)

    def attention(self, h_seq, att_w, out_w, h_dim=256, mask=None):
        '''
        h_seq: B, T, D
        att_w: linear layer
        h_dim: int D
        mask: torch.tensor(num modality,) or None
        '''
        # import pdb;pdb.set_trace()
        seq_len = h_seq.size(1)
        q = h_seq[:, -1]  # B, D
        att1 = att_w(h_seq)  # B, T, D
        q_expand = q.view(-1, 1, h_dim).contiguous().expand(-1, seq_len, -1)  # B T D
        att2 = torch.matmul(att1.reshape(-1, 1, h_dim), q_expand.reshape(-1, h_dim, 1))  # B*T 1 1
        att2 = att2.reshape(-1, seq_len)
        score = nn.functional.softmax(att2, dim=1)
        score = score.reshape(-1, seq_len, 1)  # B T 1

        # remove modalities
        if mask is not None:
            mask = mask.reshape(-1, seq_len, 1)
            score = score * mask

        res1 = torch.sum(score * h_seq, dim=1)  # B D
        res = torch.concat([res1, q], dim=1)  # B 2D
        res = out_w(res)  # B D
        res = torch.tanh(res)

        return res, score
    
    def forward(self, x, 
                mask=None):
        '''
        x: dict
        mask: torch.tensor(num modality,) or None
        '''
        if 'ego' not in x:
            if 'traj' in x:
                self.q_modality = 'traj'
            else:
                self.q_modality = 'sklt'
        for k in x:
            if k != 'ctx':
                self.encoders[k].flatten_parameters()
        if 'sklt' in x:
            B, nd, obslen, nj = x['sklt'].size()
            x['sklt'] = x['sklt'].permute(0, 2, 1, 3).reshape(B, obslen, nd*nj)  # B 2 T 17 -> B T 17 2 -> B T 2*17
        if 'ego' in x and len(x['ego'].shape) == 2:
            x['ego'] = x['ego'].unsqueeze(2)  # B T --> B T 1
        q_feat = None
        proj_feats = {}
        feats = []
        for k in x:
            if k == 'ctx':
                obs_len = x['ctx'].size(2)
                try:
                    ctx = nn.functional.interpolate(x['ctx'], size=(obs_len, 112, 112))  # B 3 T 112 112
                except:
                    print(x['ctx'].shape)
                    raise NotImplementedError()
                feat = self.encoders['ctx'](ctx)
                feat = feat.reshape(-1, FLATTEN_DIM[self.ctx_bb_nm])
                feat = self.ctx_embedder(feat)  # B C
                proj_feats[k] = self.proj[k](feat) # B C
                feat = self.ctx_sigm(feat)
            else:
                feat, _ = self.encoders[k](x[k])
                _feat = feat.permute(0, 2, 1).contiguous()  # B T C --> B C T
                proj_feats[k] = self.proj[k](self.pools[k](_feat).reshape(feat.size(0), self.h_dim))  # B C
                feat, _ = self.attention(feat, self.att_w[k], self.att_out[k], self.h_dim)
                feat = self.dropout[k](feat)
            if self.q_modality == k:
                q_feat = feat
            else:
                feats.append(feat)
        if self.pretrain:
            return proj_feats
        
        feats.append(q_feat)
        # import pdb; pdb.set_trace()
        feats = torch.stack(feats, dim=1)  # B M D
        feat_att, m_scores = self.attention(feats, self.modal_att_w, self.modal_att_out, self.h_dim, mask=mask)
        logits = {}
        for k in self.act_sets:
            logits[k] = self.final_layers[k](feat_att)
        # , m_scores.squeeze(-1)
        return {'cls_logits': logits, 'proj_feats': proj_feats}
    
    def get_pretrain_params(self):
        bb_params = []
        other_params = []
        for n, p in self.named_parameters():
            if 'encoder' in n or 'proj' in n or'embedder' in n:
                bb_params.append(p)
            else:
                other_params.append(p)
        return bb_params, other_params


if __name__ == '__main__':
    kp = torch.ones(size=[4, 2, 4, 17])
    vel = torch.ones(size=[4, 4])
    ctx = torch.ones(size=[4, 3, 4, 224, 224])
    traj = torch.ones(size=[4, 4, 4])
    x = {'traj': traj,
        'ctx': ctx,
        'sklt': kp,
        'ego': vel,
    }
    model = PCPA(ctx_bb_nm='C3D_t4_clean')
    res, _ = model(x)
    for k in res:
        print(k, res[k].shape)