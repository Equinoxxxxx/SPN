import torch

def calc_diversity_loss(protos, loss_type='triangular'):
    '''
    protos: tensor P d
    '''
    if loss_type == 'triangular':
        mask = 1 - torch.eye(protos.size(0)).to(protos.device)  # P P
        product = torch.mm(protos, protos.t())  # P P
        loss = torch.sum(mask * product) / torch.sum(mask)
    else:
        raise NotImplementedError(loss_type)

    return loss