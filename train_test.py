import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

from tools.utils import get_cls_weights_multi
from tools.loss.classification_loss import FocalLoss3
from tools.loss.mse_loss import rmse_loss, calc_mse
from tools.loss.cvae_loss import cvae_multi
from tools.metrics import calc_acc, calc_auc, calc_auc_morf, calc_confusion_matrix, calc_f1, \
    calc_mAP, calc_precision, calc_recall
from models.SGNet import target2predtraj, traj2target



def train_test_epoch(model,
                     model_name,
                     loader, 
                     optimizer=None,
                     scheduler=None,
                     batch_schedule=False,
                     warmer=None,
                     log=print, 
                     device='cuda:0',
                     modalities=None,
                     loss_params=None,
                     logsig_loss_func='margin',
                     exp_path='',
                     ):
    is_train = optimizer is not None
    if is_train:
        model.train()
        grad_req = torch.enable_grad()
    else:
        model.eval()
        grad_req = torch.no_grad()
    start = time.time()
    d_time = 0
    c_time = 0

    # get class weights
    if loss_params['cls_eff'] > 0:
        cls_weights_multi = get_cls_weights_multi(model=model,
                                                dataloader=loader,
                                                loss_weight='sklearn',
                                                device=device,
                                                use_cross=True,
                                                )
        for k in cls_weights_multi:
            if cls_weights_multi[k] is not None:
                log(k + ' class weights: ' + str(cls_weights_multi[k]))
        if loss_params['cls_loss'] == 'focal':
            focal = FocalLoss3()
    if model_name == 'sgnet' or model_name == 'sgnet_cvae':
        mse_func = rmse_loss().to(device)
    else:
        mse_func = calc_mse
    # targets and logits for whole epoch
    targets_e = {}
    logits_e = {}
    # start iteration
    b_end = time.time()
    total_logsig_loss = 0
    total_traj_mse = 0
    total_pose_mse = 0
    total_goal_loss = 0
    total_dec_loss = 0
    tbar = tqdm(loader, miniters=1)
    # loader.sampler.set_epoch(epoch)
    for n_iter, data in enumerate(tbar):
        # load inputs
        inputs = {}
        if 'img' in modalities:
            inputs['img'] = data['ped_imgs'].to(device)  # B 3 T H W
        if 'sklt' in modalities:
            inputs['sklt'] = data['obs_skeletons'].to(device)  # B 2 T nj
        if 'ctx' in modalities:
            inputs['ctx'] = data['obs_context'].to(device)
        if 'traj' in modalities:
            inputs['traj'] = data['obs_bboxes'].to(device)  # B T 4
        if 'ego' in modalities:
            inputs['ego'] = data['obs_ego'].to(device)
        if 'social' in modalities:
            inputs['social'] = data['obs_neighbor_relation'].to(device)
        
        # load gt
        targets = {}
        targets['cross'] = data['pred_act'].to(device).view(-1) # idx, not one hot
        targets['atomic'] = data['atomic_actions'].to(device).view(-1)
        targets['complex'] = data['complex_context'].to(device).view(-1)
        targets['communicative'] = data['communicative'].to(device).view(-1)
        targets['transporting'] = data['transporting'].to(device).view(-1)
        targets['age'] = data['age'].to(device).view(-1)
        targets['pred_traj'] = data['pred_bboxes'].to(device)
        targets['pred_sklt'] = data['pred_skeletons'].to(device)

        # forward
        b_start = time.time()
        with grad_req:
            logits = {}
            if model_name == 'sgnet' or model_name == 'sgnet_cvae':
                all_goal_traj, all_dec_traj = model(inputs)
                target_traj = traj2target(inputs['traj'], targets['pred_traj'])
                goal_loss = mse_func(all_goal_traj, target_traj)
                dec_loss = mse_func(all_dec_traj, target_traj)

                loss = goal_loss + dec_loss
                total_goal_loss += goal_loss.item()* inputs['traj'].size(0)
                total_dec_loss += dec_loss.item()* inputs['traj'].size(0)

                pred_traj = target2predtraj(all_dec_traj, inputs['traj'])  # B T 4
                traj_mse = calc_mse(pred_traj, targets['pred_traj'])
                total_traj_mse += traj_mse.item()
            
            # collect targets and logits in batch
            if loss_params['cls_eff']>0:
                for k in logits:
                    if n_iter == 0:
                        targets_e[k] = targets[k].detach()
                        logits_e[k] = logits[k].detach()
                    else:
                        targets_e[k] = torch.cat((targets_e[k], targets[k].detach()), dim=0)
                        logits_e[k] = torch.cat((logits_e[k], logits[k].detach()), dim=0)
            if is_train:
                # optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # display
        data_prepare_time = b_start - b_end
        b_end = time.time()
        computing_time = b_end - b_start
        d_time += data_prepare_time
        c_time += computing_time
        display_dict = {'data': data_prepare_time, 
                        'compute': computing_time,
                        'd all': d_time,
                        'c all': c_time,
                        }
        if loss_params['cls_eff'] > 0 and 'cross' in logits:
            with torch.no_grad():
                mean_logit = torch.mean(logits['cross'].detach(), dim=0)
                display_dict['logit'] = [round(logits['cross'][0, 0].item(), 4), round(logits['cross'][0, 1].item(), 4)]
                display_dict['avg logit'] = [round(mean_logit[0].item(), 4), round(mean_logit[1].item(), 4)]
        tbar.set_postfix(display_dict)
        del inputs
        if is_train:
            del loss
        torch.cuda.empty_cache()

    # calc metric
    acc_e = {}
    f1_e = {}
    f1b_e = {}
    mAP_e = {}
    prec_e = {}
    rec_e = {}
    for k in logits_e:
        acc_e[k] = calc_acc(logits_e[k], targets_e[k])
        if k == 'cross':
            f1b_e[k] = calc_f1(logits_e[k], targets_e[k], 'binary')
        f1_e[k] = calc_f1(logits_e[k], targets_e[k])
        mAP_e[k] = calc_mAP(logits_e[k], targets_e[k])
        prec_e[k] = calc_precision(logits_e[k], targets_e[k])
        rec_e[k] = calc_recall(logits_e[k], targets_e[k])
    if 'cross' in acc_e:
        auc_cross = calc_auc(logits_e['cross'], 
                             targets_e['cross'])
        conf_mat = calc_confusion_matrix(logits_e['cross'], 
                                         targets_e['cross'])
        conf_mat_norm = calc_confusion_matrix(logits_e['cross'], 
                                              targets_e['cross'], 
                                              norm='true')
    
    # return res
    res = {k:{} for k in logits_e}
    res['traj_mse'] = total_traj_mse / (n_iter+1)
    res['pose_mse'] = total_pose_mse / (n_iter+1)
    res['logsig_loss'] = total_logsig_loss / (n_iter+1)
    for k in logits_e:
        if k == 'cross':
            res[k] = {
                'acc': acc_e[k],
                'map': mAP_e[k],
                'f1': f1_e[k],
                'auc': auc_cross,
                'logits': logits_e['cross'].detach().cpu().numpy(),
            }
        else:
            res[k] = {
                'acc': acc_e[k],
                'f1': f1_e[k],
                'map': mAP_e[k],
                'logits': logits_e[k]
            }
    # log res
    log('\n')
    if loss_params['cls_eff'] > 0:
        for k in acc_e:
            if k == 'cross':
                log(f'\tacc: {acc_e[k]}\t mAP: {mAP_e[k]}\t f1: {f1_e[k]}\t f1b: {f1b_e[k]}\t AUC: {auc_cross}')
                log(f'\tprecision: {prec_e[k]}')
                log(f'\tconf mat: {conf_mat}')
                log(f'\tconf mat norm: {conf_mat_norm}')
            else:
                log(f'\t{k} acc: {acc_e[k]}\t {k} mAP: {mAP_e[k]}\t {k} f1: {f1_e[k]}')
                log(f'\t{k} recall: {rec_e[k]}')
                log(f'\t{k} precision: {prec_e[k]}')
    if loss_params['mse_eff'] > 0:
        log(f'\t mse: {total_traj_mse / (n_iter+1)}')
    if loss_params['pose_mse_eff'] > 0:
        log(f'\t pose mse: {total_pose_mse / (n_iter+1)}')
    log('\n')
    return res