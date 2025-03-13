import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import idx2onehot
import math


def calc_orth_loss_fix(protos, orth_type, threshold=1):
    '''
    protos: tensor n_p proto_dim
    '''
    orth_loss = 0
    # print(protos.size())
    protos = protos.reshape(protos.size(0), -1)
    if orth_type == 1:  # only diversity
        _mask = 1 - torch.eye(protos.size(0)).cuda()  # n_p n_p
        product = torch.matmul(protos, protos.permute(1, 0))  # n_p n_p
        orth_loss = torch.norm(_mask * product, dim=(0, 1))
    elif orth_type == 2:  # diversity and orthoganality
        _mask = torch.eye(protos.size(0)).cuda()  # n_p n_p
        product = torch.matmul(protos, protos.permute(1, 0))  # n_p n_p
        orth_loss = torch.norm(product - _mask)
    elif orth_type == 3:
        protos_ = F.normalize(protos, dim=-1)
        l2 = ((protos_.unsqueeze(-2) - protos_.unsqueeze(-1)) ** 2).sum(-1)  # np np
        neg_dis = threshold - l2
        mask = neg_dis>0
        neg_dis = neg_dis * mask.float()
        neg_dis = torch.triu(neg_dis, diagonal=1)  # upper triangle
        orth_loss = neg_dis.sum(0).sum(0)
    return orth_loss