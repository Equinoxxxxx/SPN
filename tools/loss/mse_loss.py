import torch
import torch.nn as nn
import torch.nn.functional as F


class sgnet_rmse_loss(nn.Module):
    '''
    Params:
        x_pred: (batch_size, enc_steps, dec_steps, pred_dim)
        x_true: (batch_size, enc_steps, dec_steps, pred_dim)
    Returns:
        rmse: scalar, rmse = \sum_{i=1:batch_size}()
    '''
    def __init__(self):
        super(sgnet_rmse_loss, self).__init__()
    
    def forward(self, x_pred, x_true):
        L2_diff = torch.sqrt(torch.sum((x_pred - x_true)**2, dim=3))
        # sum over prediction time steps
        L2_all_pred = torch.sum(L2_diff, dim=2)
        # mean of each frames predictions
        L2_mean_pred = torch.mean(L2_all_pred, dim=1)
        # sum of all batches
        L2_mean_pred = torch.mean(L2_mean_pred, dim=0)
        return L2_mean_pred


def calc_mse(pred:torch.Tensor, 
             target:torch.Tensor, 
             ):
    '''
    pred: ... d
    target: ... d
    '''
    _d = pred.size(-1)
    pred = pred.contiguous().view(-1, _d)
    target = target.contiguous().view(-1, _d)
    return torch.mean(torch.sqrt(torch.sum((pred-target)**2, dim=-1)))


if __name__ == '__main__':
    # loss1 = sgnet_rmse_loss()
    # loss2 = calc_mse
    # loss3 = F.mse_loss
    # x = torch.randn((32, 4, 8, 4))
    # y = torch.randn((32, 4, 8, 4))
    # l1 = loss1(x, y)
    # l2 = loss2(x, y)
    # l3 = loss3(x, y)
    # # print(l1, l1/8, l2, l3)

    # x = torch.randn((2, 8, 5, 4))
    # y = torch.randn((2, 8, 5, 4))
    # a = torch.sqrt(torch.sum((x-y)**2, dim=-1)).mean(1)  # 2 5
    # b = torch.argmin(a, dim=1)  # 2
    # print(b.shape)
    # print(b)
    # c = a[range(len(b)), b]
    # d = a[:, b]
    # print(c)
    # print(d)

    a = torch.arange(0, 6).reshape(2, 3)
    print(f'a {a}')
    print(a[:, [0,2]])
    print(a[[0,1]])
    print(a[[0,1], [0,1,2]])