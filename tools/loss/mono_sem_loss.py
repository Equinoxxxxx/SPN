import torch
import torch.nn.functional as F

def get_topk_mono_scale(proto_simi, k):
    '''
    proto_simi: B n_proto
    k: int
    '''
    proto_simi = torch.sigmoid(proto_simi)
    proto_simi = proto_simi / proto_simi.sum(dim=1, keepdim=True)
    topk = torch.topk(proto_simi, k, dim=1)[1]
    return topk

def calc_topk_monosem(proto_simi:torch.Tensor, 
                      k:int,
                      topk_metric:str='relative_var'):
    '''
    proto_simi: B n_proto
    k: int
    '''
    # calc mean and var
    mean = proto_simi.mean(dim=0)  # (P)
    var = proto_simi.var(dim=0, unbiased=True)  # (P)
    # relative var
    relative_variance = (proto_simi - mean)**2 / (var.unsqueeze(0) + 1e-5)  # (B, P)
    # topk
    if topk_metric == 'relative_var':
        top_k_values, top_k_indices = torch.topk(relative_variance, k, dim=0)  # (k, P) (k, P)
    elif topk_metric == 'activation':
        _, top_k_indices = torch.topk(proto_simi, k, dim=0)  # (k, P)
        top_k_values = torch.gather(relative_variance, 0, top_k_indices)  # (k, P)
    else:
        raise ValueError(topk_metric)
    # sparsity value
    sparsity = top_k_values.mean(dim=0)  # (k, P) -> (P)

    return sparsity, top_k_indices


def calc_mono_sem_align_loss(weights: torch.Tensor,
                        sparsity: torch.Tensor,
                        loss_func: str='cosine_simi'):
    '''
    weights: n_cls, n_proto
    sparsity: n_proto,
    '''
    if loss_func == 'cosine_simi':
        # sum for all classes
        weights_sum = weights.sum(dim=0)  # (n_proto,)
        cos_sim = F.cosine_similarity(weights_sum, sparsity, dim=0)
        loss = 1 - cos_sim
    elif loss_func == 'multiply_l2':
        weighted_weights = weights * sparsity.unsqueeze(0)  # (n_cls, n_proto)
        loss = weighted_weights.norm(2)  # scalar
    else:
        raise ValueError(loss_func)
    
    return loss