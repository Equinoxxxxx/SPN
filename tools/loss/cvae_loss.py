import torch

def sgnet_cvae_loss(pred_traj, target, first_history_index = 0):
    '''
    CVAE loss use best-of-many
    pred_traj: B obslen predlen K 4
    '''
    K = pred_traj.shape[3]
    
    target = target.unsqueeze(3).repeat(1, 1, 1, K, 1)
    total_loss = []
    for enc_step in range(first_history_index, pred_traj.size(1)):
        traj_rmse = torch.sqrt(torch.sum((pred_traj[:,enc_step,:,:,:] - target[:,enc_step,:,:,:])**2, dim=-1)).sum(dim=1)  # B K
        best_idx = torch.argmin(traj_rmse, dim=1)
        loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
        total_loss.append(loss_traj)
    
    return sum(total_loss)/len(total_loss)


def calc_stoch_mse(pred:torch.Tensor, 
                   target:torch.Tensor,
                   stoch_mse_type='avg'):
    '''
    pred: B predlen K ... d
    target: B predlen ... d
    '''
    K = pred.shape[2]
    target =  target.unsqueeze(2)  # B predlen 1 ... d
    try:
        mse_all = torch.sqrt(torch.sum((pred-target)**2, dim=-1)).mean(1)  # B K ...
    except:
        import pdb;pdb.set_trace()
    while len(mse_all.shape) > 2:
        mse_all = mse_all.mean(-1)  # B K
    if stoch_mse_type == 'best':
        best_idx = torch.argmin(mse_all, dim=1)  # B 
        mse = mse_all[range(len(best_idx)), best_idx]
    elif stoch_mse_type == 'avg':
        mse = mse_all.mean(1)  # B
    return mse



