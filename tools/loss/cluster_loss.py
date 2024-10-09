import torch
import torch.nn as nn
from torch.nn import functional as F


def calc_batch_simi_simple(feat_dict, 
                           log_logit_scale, 
                           simi_func='dot_prod',
                           pair_mode='pair_wise'):
    '''
    feat_dict: dict{k: tensor(B, c)}
    '''
    eps = 1e-7
    logit_scale = log_logit_scale.exp()
    simi_mats = []
    modalities = list(feat_dict.keys())
    n_m = len(modalities)
    if simi_func == 'dot_prod' and 'pair_wise' in pair_mode:
        # traverse modality dim
        for i in range(n_m):
            mi = modalities[i]
            for j in range(i+1, n_m):
                mj = modalities[j]
                zi = feat_dict[mi]  # b, c
                zj = feat_dict[mj]  # b, c
                # cosine simi
                zi_norm, zj_norm = \
                    zi/(zi.norm(dim=1, keepdim=True)+eps), zj/(zj.norm(dim=1, keepdim=True)+eps)
                simi_mat1 = logit_scale * zi_norm @ zj_norm.t() + eps
                simi_mat2 = simi_mat1.t()
                simi_mats += [simi_mat1, simi_mat2]
    else:
        raise NotImplementedError(simi_func, pair_mode)
    
    return simi_mats  # list: n_pair*[b, b]


def calc_contrast_loss(simi_mats, pair_mode):
    '''
    simi_mats: list [tensor(b, b)]
    '''
    if pair_mode == 'pair_wise_norm_orth' or\
        pair_mode == 'pair_wise_orth':
        loss = 0
        simi_mats = torch.stack(simi_mats, dim=0)  # n_pair, b, b
        I_ = torch.unsqueeze(torch.eye(simi_mats.size(1)), dim=0).to(simi_mats.device)  # 1, b, b
        orth_loss = torch.mean(torch.norm(simi_mats - I_))
        return orth_loss
    else:
        assert pair_mode in ('pair_wise', 
                                 'proto_pair_wise', 
                                 'bridge',
                                 'proto_bridge')
        ce = 0
        n_pairs = len(simi_mats)
        # print('n pairs', n_pairs)
        label = torch.arange(simi_mats[0].size(0), device=simi_mats[0].device, dtype=torch.long)  # b
        i=0
        for mat in simi_mats:  # b, b
            # print('cur pair', i)
            # print('mat', mat)
            # print('label', label)
            ce = ce + F.cross_entropy(mat, label)
            i+=1
    
        return ce / len(simi_mats)