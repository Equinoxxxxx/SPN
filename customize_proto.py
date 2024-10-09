import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

from tools.utils import get_cls_weights_multi
from tools.loss.classification_loss import FocalLoss3
from tools.loss.mse_loss import sgnet_rmse_loss, calc_mse
from tools.loss.cvae_loss import sgnet_cvae_loss, calc_stoch_mse
from tools.loss.mono_sem_loss import calc_topk_monosem, calc_mono_sem_align_loss
from tools.loss.diversity_loss import calc_diversity_loss
from tools.metrics import calc_acc, calc_auc, calc_confusion_matrix, calc_f1, \
    calc_mAP, calc_precision, calc_recall


def customize_proto(args,
                    model_parallel
                    ):
    P = model_parallel.module.proto_enc.weight.shape[0]
    if args.proto_value_to_rank == 'abs_weight':
        weights = torch.zeros(P).to(model_parallel.module.proto_enc.weight.device)
        for act_set in model_parallel.module.proto_dec:
            cur_weight = model_parallel.module.proto_dec[act_set].weight # n_cls, P
            weights += cur_weight.abs().sum(dim=0) # P
    elif args.proto_value_to_rank == 'sparsity':
        weights = model_parallel.module.all_sparsity # P
    else:
        raise NotImplementedError(args.proto_value_to_rank)
    if args.proto_rank_criteria == 'num_select':
        _, selected_idx = weights.topk(args.proto_num_select)
    else:
        raise NotImplementedError(args.proto_rank_criteria)
    # remove rest proto
    with torch.no_grad():
        for i in range(P):
            if i not in selected_idx:
                model_parallel.module.proto_enc.weight[i].zero_()
    
    return model_parallel
