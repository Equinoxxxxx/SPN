import math
import torch
from torch import nn
import numpy as np
from tools.datasets.TITAN import ACT_SET_TO_N_CLS
    

class PedGraph(nn.Module):
    def __init__(self, 
                 modalities=['sklt', 'ctx', 'ego'],
                 act_sets=['cross'], 
                 seg=True, 
                 h3d=False, 
                 nodes=17, 
                 proj_norm='bn',
                 proj_actv='leakyrelu',
                 pretrain=True,
                 n_mlp=1,
                 proj_dim=256,
                 ):
        super(PedGraph, self).__init__()
        self.model_name = 'ped_graph'
        self.modalities = modalities
        self.h3d = h3d # bool if true 3D human keypoints data is enable otherwise 2D is only used
        self.seg = seg
        self.act_sets = act_sets
        self.proj_norm = proj_norm
        self.proj_actv = proj_actv
        self.pretrain = pretrain
        self.n_mlp = n_mlp
        self.ch = 3 if h3d else 2
        self.ch1, self.ch2 = 32, 64
        i_ch = 4 if self.seg else 3
        self.proj_dim = proj_dim if proj_dim > 0 else self.ch2

        # init contrast scale factor
        self.logit_scale = nn.parameter.Parameter(
            torch.ones([]) * np.log(1 / 0.07))
        
        self.data_bn = nn.BatchNorm1d(self.ch * nodes)
        bn_init(self.data_bn, 1)
        self.drop = nn.Dropout(0.25)
        A = np.stack([np.eye(nodes)] * 3, axis=0)
        
        if 'ctx' in self.modalities:
            self.ctx_encoder0 = nn.Sequential(
                nn.Conv2d(i_ch, self.ch1, kernel_size=3, stride=1, padding=0, bias=False), 
                nn.BatchNorm2d(self.ch1), 
                nn.SiLU())
            self.ctx_encoder1 = nn.Sequential(
                nn.Conv2d(self.ch1, self.ch1, kernel_size=3, stride=1, padding=0, bias=False), 
                nn.BatchNorm2d(self.ch1), 
                nn.SiLU())
            self.ctx_encoder2 = nn.Sequential(
                nn.Conv2d(self.ch1, self.ch2, kernel_size=2, stride=1, padding=0, bias=False), 
                nn.BatchNorm2d(self.ch2), 
                nn.SiLU())
            self.ctx_proj_gap = nn.AdaptiveAvgPool2d(1)
            ctx_proj = []
            for i in range(self.n_mlp):
                if i == 0:
                    ctx_proj.append(nn.Linear(self.ch2, self.proj_dim))
                else:
                    ctx_proj.append(nn.Linear(self.ch2, self.proj_dim))
                if self.proj_norm == 'ln':
                    ctx_proj.append(nn.LayerNorm(self.proj_dim))
                elif self.proj_norm == 'bn':
                    ctx_proj.append(nn.BatchNorm1d(self.proj_dim))
                if self.proj_actv == 'silu':
                    ctx_proj.append(nn.SiLU())
                elif self.proj_actv == 'relu':
                    ctx_proj.append(nn.ReLU())
                elif self.proj_actv == 'leakyrelu':
                    ctx_proj.append(nn.LeakyReLU())
            if self.n_mlp > 1:
                ctx_proj.append(nn.Linear(self.proj_dim, self.proj_dim))
            self.ctx_proj = nn.Sequential(
                *ctx_proj
            )
        if 'ego' in self.modalities:
            self.ego_encoder0 = nn.Sequential(
                nn.Conv1d(1, self.ch1, 2, bias=False),   # kernel: 3 --> 2
                nn.BatchNorm1d(self.ch1), 
                nn.SiLU())
            self.ego_encoder1 = nn.Sequential(
                nn.Conv1d(self.ch1, self.ch1, 2, bias=False),    # kernel: 3 --> 2
                nn.BatchNorm1d(self.ch1), 
                nn.SiLU())
            self.ego_encoder2 = nn.Sequential(
                nn.Conv1d(self.ch1, self.ch2, kernel_size=2, bias=False), 
                nn.BatchNorm1d(self.ch2), 
                nn.SiLU())
            self.ego_proj_gap = nn.AdaptiveAvgPool1d(1)
            ego_proj = []
            for i in range(self.n_mlp):
                if i == 0:
                    ego_proj.append(nn.Linear(self.ch2, self.proj_dim))
                else:
                    ego_proj.append(nn.Linear(self.proj_dim, self.proj_dim))
                if self.proj_norm == 'ln':
                    ego_proj.append(nn.LayerNorm(self.proj_dim))
                elif self.proj_norm == 'bn':
                    ego_proj.append(nn.BatchNorm1d(self.proj_dim))
                if self.proj_actv == 'silu':
                    ego_proj.append(nn.SiLU())
                elif self.proj_actv == 'relu':
                    ego_proj.append(nn.ReLU())
                elif self.proj_actv == 'leakyrelu':
                    ego_proj.append(nn.LeakyReLU())
            if self.n_mlp > 1:
                ego_proj.append(nn.Linear(self.proj_dim, self.proj_dim))
            self.ego_proj = nn.Sequential(
                *ego_proj
            )
        # ----------------------------------------------------------------------------------------------------
        self.sklt_encoder1 = TCN_GCN_unit(self.ch, self.ch1, A, residual=False)
        self.sklt_encoder2 = TCN_GCN_unit(self.ch1, self.ch2, A)
        self.gap = nn.AdaptiveAvgPool2d(1)
        sklt_proj = []
        for i in range(self.n_mlp):
            if i == 0:
                sklt_proj.append(nn.Linear(self.ch2, self.proj_dim))
            else:
                sklt_proj.append(nn.Linear(self.proj_dim, self.proj_dim))
            if self.proj_norm == 'ln':
                sklt_proj.append(nn.LayerNorm(self.proj_dim))
            elif self.proj_norm == 'bn':
                sklt_proj.append(nn.BatchNorm1d(self.proj_dim))
            if self.proj_actv == 'silu':
                sklt_proj.append(nn.SiLU())
            elif self.proj_actv == 'relu':
                sklt_proj.append(nn.ReLU())
            elif self.proj_actv == 'leakyrelu':
                sklt_proj.append(nn.LeakyReLU())
        if self.n_mlp > 1:
            sklt_proj.append(nn.Linear(self.proj_dim, self.proj_dim))
        self.sklt_proj = nn.Sequential(
            *sklt_proj
        )
        # ----------------------------------------------------------------------------------------------------
        # self.l3 = TCN_GCN_unit(self.ch2, self.ch2, A)

        # if frames:
        #     self.conv3 = nn.Sequential(
        #         nn.Conv2d(self.ch2, self.ch2, kernel_size=2, stride=1, padding=0, bias=False), 
        #         nn.BatchNorm2d(self.ch2), nn.SiLU())
            
        # if vel:
        #     self.v3 = nn.Sequential(
        #         nn.Conv1d(self.ch2, self.ch2, kernel_size=2, bias=False), 
        #         nn.BatchNorm1d(self.ch2), nn.SiLU())
        # ----------------------------------------------------------------------------------------------------
        
        self.att = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.ch2, self.ch2, bias=False),
            nn.BatchNorm1d(self.ch2), 
            nn.Sigmoid()
        )
        self.last_layers = {}
        for act_set in self.act_sets:
            self.last_layers[act_set] = nn.Linear(self.ch2, ACT_SET_TO_N_CLS[act_set])
            nn.init.normal_(self.last_layers[act_set].weight, 0, math.sqrt(2. / ACT_SET_TO_N_CLS[act_set]))
        self.last_layers = nn.ModuleDict(self.last_layers)
        # pooling sigmoid fucntion for image feature fusion
        self.ctx_sigm = nn.Sigmoid()
        if 'ego' in self.modalities:
            self.ego_sigm = nn.Sigmoid()
    
    def forward(self, x): 
        kp = x.get('sklt')
        frame = x.get('ctx')  # b 4 h w
        if len(frame.shape) == 5:  # b 4 t h w
            frame = frame[:, :, -1]
        B, C, T, V = kp.shape  # b, 2, t, 17
        kp = kp.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)  # b 2*17, t
        kp = self.data_bn(kp)
        kp = kp.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()
        if 'ego' in x:
            vel = x.get('ego')  # b t
            vel = vel.view(B, 1, T)  # b 1 t
        
        if 'ctx' in self.modalities:
            f1 = self.ctx_encoder0(frame) 
        if 'ego' in self.modalities:
            v1 = self.ego_encoder0(vel)

        # --------------------------
        x1 = self.sklt_encoder1(kp)
        if 'ctx' in self.modalities:
            f1 = self.ctx_encoder1(f1)
            f1_gap = self.ctx_proj_gap(f1)
            f1_sigm = self.ctx_sigm(f1_gap)
            x1 = x1 * f1_sigm
        if 'ego' in self.modalities:
            v1 = self.ego_encoder1(v1)
            v1_gap = self.ego_proj_gap(v1)
            v1_sigm = self.ego_sigm(v1_gap)
            x1 = x1 * v1_sigm.unsqueeze(-1)
        # --------------------------
        
        # --------------------------
        x1 = self.sklt_encoder2(x1)
        x_fuse = x1
        if 'ctx' in self.modalities:
            f1 = self.ctx_encoder2(f1)
            f1_gap = self.ctx_proj_gap(f1)
            f1_sigm = self.ctx_sigm(f1_gap)
            x_fuse = x_fuse * f1_sigm
        if 'ego' in self.modalities:  
            v1 = self.ego_encoder2(v1)
            v1_gap = self.ego_proj_gap(v1)
            v1_sigm = self.ego_sigm(v1_gap)
            x_fuse = x_fuse * v1_sigm.unsqueeze(-1)
        # --------------------------
        # x1 = self.l3(x1)
        # if self.frames:
        #     f1 = self.conv3(f1) 
        #     x1 = x1.mul(self.pool_sigm_2d(f1))
        # if self.vel:  
        #     v1 = self.v3(v1)
        #     x1 = x1.mul(self.pool_sigm_1d(v1).unsqueeze(-1))
        # --------------------------
        
        x_gap = self.sklt_proj(self.gap(x1).view(B, self.ch2))
        feats = {'sklt': x_gap}
        if 'ctx' in self.modalities:
            f1_gap = self.ctx_proj(f1_gap.view(B, self.ch2))
            feats['ctx'] = f1_gap
        if 'ego' in self.modalities:
            v1_gap = self.ego_proj(v1_gap.view(B, self.ch2))
            feats['ego'] = v1_gap
        if self.pretrain:
            return feats
        
        x_fuse = self.gap(x_fuse).view(B, self.ch2)  # b C t 17 --> b C
        x_fuse = self.att(x_fuse).mul(x_fuse) + x_fuse
        x_fuse = self.drop(x_fuse)
        logits = {}
        for act_set in self.act_sets:
            logits[act_set] = self.last_layers[act_set](x_fuse)

        return {'cls_logits': logits,
                'proj_feats': feats,
                }
    
    def get_pretrain_params(self):
        bb_params = []
        other_params = []
        for n, p in self.named_parameters():
            if 'encoder' in n or 'proj' in n:
                bb_params.append(p)
            else:
                other_params.append(p)
        return bb_params, other_params

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            self.A = torch.autograd.Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)
    
    def L2_norm(self, A):
        # A:N,V,V
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            A = self.L2_norm(A)
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):

            A1 = A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True):
        super(TCN_GCN_unit, self).__init__()
        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if self.residual:
            self.residual_module = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        if not self.residual:
            y = self.relu(self.tcn1(self.gcn1(x)))
        elif (self.in_channels == self.out_channels) and (self.stride == 1):
            y = self.relu(self.tcn1(self.gcn1(x)) + x)
        else:
            y = self.relu(self.tcn1(self.gcn1(x)) + self.residual_module(x))
        return y


if __name__ == '__main__':
    kp = torch.ones(size=[4, 2, 4, 17])
    vel = torch.ones(size=[4, 4])
    ctx = torch.ones(size=[4, 4, 4, 48, 48])
    x = {
        'ctx': ctx,
        'sklt': kp,
        'ego': vel,
    }
    model = PedGraph()
    print(model(x))