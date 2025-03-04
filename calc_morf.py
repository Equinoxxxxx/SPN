import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import cv2
import copy
import argparse
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from get_args import get_args, process_args
from tools.utils import makedir, write_info_txt
from main import construct_data_loader, construct_model
from tools.log import create_logger
from explain import select_topk
from explain_no_fusion import select_topk_no_fusion
from train_test import train_test_epoch
from customize_proto import customize_proto


def calc_morf():
    parser = argparse.ArgumentParser(description='explain args')
    parser.add_argument('--args_path', type=str, 
        default='/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp726/args.pkl')
    parser.add_argument('--ckpt_path', type=str, 
        default='/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp726/ckpt/22_0.0000.pth')
    # parser.add_argument('--dataset_name', type=str, default='TITAN')
    morf_args = parser.parse_args([])

    # load args
    init_args = get_args()
    init_args = process_args(init_args)
    init_args_dict = vars(init_args)
    with open(morf_args.args_path, 'rb') as f:
        args = pickle.load(f)
    args_dict = vars(args)
    extra_args = {key: init_args_dict[key] for key in init_args_dict if key not in args_dict}
    args_dict.update(extra_args)
    args = argparse.Namespace(**args_dict)
    args.test_customized_proto = True
    args.batch_size = [1,1]
    if not hasattr(args, 'max_n_neighbor'):
        args.max_n_neighbor = 10
    # log
    ckpt_epoch = int(morf_args.ckpt_path.split('/')[-1].split('_')[0])
    exp_dir = morf_args.args_path.replace('args.pkl', '')
    morf_root = os.path.join(exp_dir, f'morf_epoch_{ckpt_epoch}' )
    makedir(morf_root)
    log, logclose = create_logger(log_filename=os.path.join(morf_root, 'morf.log'))
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load the data
    log('----------------------------Construct data loaders-----------------------------')
    train_loaders, val_loaders, test_loaders = construct_data_loader(args, log)
    test_loaders = test_loaders[0]
    # construct the model
    log('----------------------------Construct model-----------------------------')
    model = construct_model(args, device)
    model = model.float().to(device)
    state_dict = torch.load(morf_args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model_parallel = torch.nn.parallel.DataParallel(model)
    model_parallel.eval()
    morf_dict = {}
    for act_set in args.act_sets:
        relevs = model.module.proto_dec[act_set].weight.detach()  # n_cls, n_proto
        ori_weights = copy.deepcopy(relevs)
        if act_set == 'cross':
            _relevs = torch.zeros_like(relevs)
            _relevs[0] = copy.deepcopy(relevs[0]) - copy.deepcopy(relevs[1])
            _relevs[1] = copy.deepcopy(relevs[1]) - copy.deepcopy(relevs[0])
            relevs = _relevs
        for loader in test_loaders:
            dataset_name = loader.dataset.dataset_name
            log(f'--------------------------Calc MORF for {act_set} on {dataset_name}---------------------------')
            # morf iters
            print(f'max relev: {torch.max(relevs)}, min relev: {torch.min(relevs)}')
            nf = relevs.size(-1)
            morf_logits = torch.zeros(size=(relevs.size(0), loader.dataset.num_samples, nf+1))  # c B nf
            print(morf_logits.size())
            # class iter
            for c in tqdm(range(relevs.size(0)), desc='class iter'):
                relev_c = relevs[c].unsqueeze(0)  # 1 n_feat
                # get masks for cur class
                morf_masks = get_batch_morf_masks(relev_c).to(device)  # 1, nf+1, nf
                # nf iter
                for f in tqdm(range(nf+1), desc='nf iter'):
                    mask = morf_masks[0, f]  # 1 nf
                    # masked = ori_weights * mask  #  n_cls n_feat
                    for i_iter, data in enumerate(loader):
                        # load inputs
                        inputs = {}
                        if 'img' in args.modalities:
                            inputs['img'] = data['ped_imgs'].to(device)  # B 3 T H W
                        if 'sklt' in args.modalities:
                            inputs['sklt'] = data['obs_skeletons'].to(device)  # B 2 T nj
                        if 'ctx' in args.modalities:
                            inputs['ctx'] = data['obs_context'].to(device)
                        if 'traj' in args.modalities:
                            inputs['traj'] = data['obs_bboxes'].to(device)  # B T 4
                        if 'ego' in args.modalities:
                            inputs['ego'] = data['obs_ego'].to(device)
                        if 'social' in args.modalities:
                            inputs['social'] = data['obs_neighbor_relation'].to(device)
                        
                        # load gt
                        targets = {}
                        targets['cross'] = data['pred_act'].to(device).view(-1) # idx, not one hot
                        targets['atomic'] = data['atomic_actions'].to(device).view(-1)
                        targets['simple'] = data['simple_context'].to(device).view(-1)
                        targets['complex'] = data['complex_context'].to(device).view(-1)
                        targets['communicative'] = data['communicative'].to(device).view(-1)
                        targets['transporting'] = data['transporting'].to(device).view(-1)
                        targets['age'] = data['age'].to(device).view(-1)
                        targets['pred_traj'] = data['pred_bboxes'].to(device)  # B predlen 4
                        if 'sklt' in args.modalities:
                            targets['pred_sklt'] = data['pred_skeletons'].to(device)  # B ndim predlen nj

                        # forward
                        with torch.no_grad():
                            batch = (inputs, targets)
                            out = model(batch, is_train=0, mask=mask)
                            logit = out['cls_logits'][act_set].detach() # 1 n_cls
                            logit = F.softmax(logit, dim=-1)
                            morf_logits[c, i_iter, f] = logit[0, c]
            morf_logits = morf_logits.mean(1)  # c, nf
            morf_logits_path = os.path.join(morf_root, f'morf_logits_{act_set}_{dataset_name}.pkl')
            with open(morf_logits_path, 'wb') as f:
                pickle.dump(morf_logits, f)
            # calc auc
            auc_morf = torch.zeros(relevs.size(0))
            auc_morf_max_norm = torch.zeros(relevs.size(0))
            auc_morf_max_min_norm = torch.zeros(relevs.size(0))
            for c in range(relevs.size(0)):
                curve_path = os.path.join(morf_root, f'morf_curve_{act_set}_{dataset_name}_cls{c}.png')
                plot_morf(morf_logits[c], curve_path)
                auc_morf[c] = calc_auc_morf(morf_logits[c])
                print(torch.max(morf_logits[c]), torch.min(morf_logits[c]))
                auc_morf_max_norm[c] = auc_morf[c] / torch.max(morf_logits[c])
                # auc_morf_norm[c] = (auc_morf[c] - torch.min(morf_logits[c])) / (torch.max(morf_logits[c]) - torch.min(morf_logits[c]))
            log(f'{act_set} {dataset_name} auc-morf: {auc_morf} \tauc-morf max norm: {auc_morf_max_norm}')
            log(f'Res saved in {morf_root}')
            logclose()

def get_one_feat_masks(n_feat):
    '''
    n_feat: int, num of features
    '''
    return torch.eye(n_feat)


def get_batch_morf_masks(relevs):
    '''
    relevs: torch.tensor(b(1), n_feat)
    return:
        masks: torch.tensor(b(1), n_feat+1, n_feat)
    '''
    b_size = relevs.size(0)
    n_feat = relevs.size(1)
    neg_masks = (relevs<0)  # b, n_feat
    importances = torch.abs(relevs)
    idcs = torch.argsort(importances, dim=-1, descending=True)  # b, n_feat
    # nn_idcs = idcs.repeat(n_feat, 1, 1).permute(1, 0, 2)  # b, n_feat, n_feat
    # nn = torch.arange(n_feat).repeat(n_feat)  # n_feat, n_feat
    # tril_idcs = torch.tril(nn_idcs, diagonal=0)  # b, n_feat, n_feat
    # tril_idcs += torch.triu(-1*torch.ones(size=(n_feat, n_feat)), diagonal=1)  # lower triangle with rest part -1

    masks = torch.ones(size=(b_size, n_feat+1, n_feat))
    for i in range(b_size):
        for j in range(n_feat):
            masks[i, j+1, idcs[i, :j+1]] = 0
    # turn over masks of neg relev
    masks = masks.permute(0, 2, 1)
    masks[neg_masks] -= 1
    masks[neg_masks] *= -1

    return masks.permute(0, 2, 1)

def plot_morf(logits, path):
    '''
    logits: torch.tensor(n,) in descending order
    '''
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    logits = logits.detach().cpu().numpy()
    plt.plot(logits)
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Prediction')
    plt.savefig(path)
    plt.close()

def calc_auc_morf(logits):
    '''
    logits: torch.tensor(n, )
    '''
    with torch.no_grad():
        n = logits.size(0)
        res = 0
        for i in range(n-1):
            res += (logits[i] + logits[i+1]) / 2
        res = res / n
    return res.detach().cpu()