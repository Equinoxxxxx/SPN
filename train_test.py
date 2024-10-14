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
from tools.loss.cluster_loss import calc_batch_simi_simple, calc_contrast_loss
from tools.metrics import calc_acc, calc_auc, calc_confusion_matrix, calc_f1, \
    calc_mAP, calc_precision, calc_recall
from models.SGNet import accumulate_traj, traj_to_sgnet_target


def train_test_epoch(args,
                     model,
                     model_name,
                     dataloader, 
                     optimizer=None,
                     scheduler=None,
                     batch_schedule=False,
                     warmer=None,
                     log=print, 
                     device='cuda:0',
                     modalities=None,
                     loss_params=None,
                     ):
    is_train = optimizer is not None
    if is_train:
        model.train()
        grad_req = torch.enable_grad()
        cur_lr = []
        for param_groups in optimizer.state_dict()['param_groups']:
            cur_lr.append(param_groups['lr'])
        log(f'cur lr: {cur_lr}')
    else:
        model.eval()
        grad_req = torch.no_grad()
    start = time.time()
    d_time = 0
    c_time = 0

    # get class weights and classification loss func
    if loss_params['cls_eff'] > 0:
        cls_weights_multi = get_cls_weights_multi(model=model,
                                                dataloader=dataloader,
                                                loss_weight='sklearn',
                                                device=device,
                                                act_sets=args.act_sets,
                                                )
        for k in cls_weights_multi:
            if cls_weights_multi[k] is not None:
                log(k + ' class weights: ' + str(cls_weights_multi[k]))
        if loss_params['cls_loss_func'] == 'focal':
            cls_loss_func = FocalLoss3()
        else:
            cls_loss_func = F.cross_entropy  # input, target
    
    # get regression loss func
    if loss_params['mse_eff'] > 0:
        if model_name == 'sgnet' or model_name == 'sgnet_cvae':
            traj_loss_func = sgnet_rmse_loss().to(device)
        else:
            traj_loss_func = calc_mse
    if loss_params['pose_mse_eff'] > 0:
        if model_name == 'deposit':
            pass
    # init loss and results for the whole epoch
    total_traj_mse = 0
    total_pose_mse = 0
    total_pose_loss = 0
    total_traj_loss = 0
    total_sgnet_goal_loss = 0
    total_sgnet_dec_loss = 0
    total_sgnet_cvae_loss = 0
    total_logsig_loss = 0
    total_diversity_loss = 0
    total_mono_sem_loss = 0
    total_mono_sem_l1_loss = 0
    total_mono_sem_align_loss = 0
    total_batch_sparsity = 0
    total_cluster_loss = 0
    all_proto_simi = []
    # targets and logits for whole epoch
    targets_e = {}
    logits_e = {}
    # start iteration
    b_end = time.time()
    tbar = tqdm(dataloader, miniters=1)
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
        targets['simple'] = data['simple_context'].to(device).view(-1)
        targets['complex'] = data['complex_context'].to(device).view(-1)
        targets['communicative'] = data['communicative'].to(device).view(-1)
        targets['transporting'] = data['transporting'].to(device).view(-1)
        targets['age'] = data['age'].to(device).view(-1)
        targets['pred_traj'] = data['pred_bboxes'].to(device)  # B predlen 4
        targets['pred_sklt'] = data['pred_skeletons'].to(device)  # B ndim predlen nj

        # forward
        b_start = time.time()
        with grad_req:
            logits = {}
            if model_name == 'sgnet':
                out = model(inputs)
                pred_traj = out['pred_traj']
                all_goal_traj, all_dec_traj = out['ori_output']
                target_traj = traj_to_sgnet_target(inputs['traj'], targets['pred_traj'])
                goal_loss = traj_loss_func(all_goal_traj, target_traj)
                dec_loss = traj_loss_func(all_dec_traj, target_traj)
                # pred_traj = accumulate_traj(inputs['traj'], all_dec_traj)  # B predlen 4
                traj_mse = calc_mse(pred_traj, targets['pred_traj'])
                loss = goal_loss + dec_loss
                # add to total loss
                total_sgnet_goal_loss += goal_loss.item()
                total_sgnet_dec_loss += dec_loss.item()
                total_traj_mse += traj_mse.item()
            elif model_name == 'sgnet_cvae':
                gt_traj = targets['pred_traj']
                target_traj = traj_to_sgnet_target(inputs['traj'], gt_traj)
                out = model(inputs, training=is_train, targets=target_traj)
                pred_traj = out['pred_traj']
                all_goal_traj, cvae_dec_traj, KLD_loss, _  = out['ori_output']
                goal_loss = traj_loss_func(all_goal_traj, target_traj)
                cvae_loss = sgnet_cvae_loss(cvae_dec_traj, target_traj)
                goal_loss = traj_loss_func(all_goal_traj, target_traj)
                # pred_traj = accumulate_traj(inputs['traj'], all_dec_traj)  # B predlen K 4
                traj_mse = calc_stoch_mse(pred_traj,  # B predlen K 4
                                          gt_traj, # B predlen 4
                                          loss_params['stoch_mse_type'])
                loss = goal_loss + cvae_loss + KLD_loss.mean()
                # add to total loss
                total_sgnet_goal_loss += goal_loss.item()
                total_sgnet_dec_loss += cvae_loss.item()
                total_traj_mse += traj_mse.mean().item()
            elif model_name == 'next':
                out = model(inputs)
                pred_traj = out['pred_traj']
                logits = out['cls_logits']
                traj_mse = calc_mse(pred_traj, targets['pred_traj'])
                loss = traj_mse * loss_params['mse_eff']
                if loss_params['cls_eff'] > 0:
                    for k in logits:
                        if n_iter == 0:
                            targets_e[k] = targets[k].detach()
                            logits_e[k] = logits[k].detach()
                        else:
                            targets_e[k] = torch.cat((targets_e[k], targets[k].detach()), dim=0)
                            logits_e[k] = torch.cat((logits_e[k], logits[k].detach()), dim=0)
                    ce_dict = {}
                    for k in logits:
                        ce_dict[k] = cls_loss_func(logits[k], 
                                                   targets[k], 
                                                   weight=cls_weights_multi[k])
                        loss = loss + ce_dict[k] * loss_params['cls_eff']
                total_traj_mse += traj_mse.item()
            elif model_name == 'deposit':
                batch = (inputs, targets)
                out = model(batch, loss_params['n_sampling'], is_train)
                loss = out['loss'].mean()
                # predicted pose
                pred_pose = out['pred']  # B K ndim*nj T
                gt_pose = targets['pred_sklt']  # B ndim predlen nj
                batch_size, n_dim, pred_len, nj = gt_pose.size()
                pred_pose = pred_pose.reshape(batch_size, -1, n_dim, nj, pred_len).\
                            permute(0,4,1,3,2)
                gt_pose = gt_pose.permute(0,2,3,1)
                pose_mse = calc_stoch_mse(pred_pose,  # B predlen K nj ndim
                                            gt_pose,  # B predlen nj ndim
                                            loss_params['stoch_mse_type'])
                total_pose_mse += pose_mse.mean().item()
                total_pose_loss += loss.item()
            elif model_name in ('PCPA', 'ped_graph'):
                out = model(inputs)
                loss = 0
                try:
                    logits = out['cls_logits']
                except:
                    import pdb; pdb.set_trace()
                if loss_params['cls_eff'] > 0:
                    for k in logits:
                        if n_iter == 0:
                            targets_e[k] = targets[k].detach()
                            logits_e[k] = logits[k].detach()
                        else:
                            targets_e[k] = torch.cat((targets_e[k], targets[k].detach()), dim=0)
                            logits_e[k] = torch.cat((logits_e[k], logits[k].detach()), dim=0)
                    ce_dict = {}
                    for k in logits:
                        ce_dict[k] = cls_loss_func(logits[k], 
                                                   targets[k], 
                                                   weight=cls_weights_multi[k])
                        loss = loss + ce_dict[k] * loss_params['cls_eff']
            elif model_name == 'pedspace':
                batch = (inputs, targets)
                out = model(batch, is_train)
                logits = out['cls_logits']
                loss = 0
                # pose loss
                if loss_params['pose_mse_eff'] > 0:
                    pose_loss = out['pose_loss'].mean()
                    total_pose_loss += pose_loss.item()
                    loss = loss + pose_loss * loss_params['pose_mse_eff']
                    # predicted pose
                    pred_pose = out['pred_pose']  # B K ndim*nj obslen+predlen
                    gt_pose = targets['pred_sklt']  # B ndim predlen nj
                    batch_size, n_dim, pred_len, nj = gt_pose.size()
                    obslen = pred_pose.size(3) - pred_len
                    pred_pose = pred_pose[:,:,:,obslen:].reshape(batch_size, -1, n_dim, nj, pred_len).\
                                permute(0,4,1,3,2)
                    gt_pose = gt_pose.permute(0,2,3,1)
                    pose_mse = calc_stoch_mse(pred_pose,  # B predlen K nj ndim
                                                gt_pose,  # B predlen nj ndim
                                                loss_params['stoch_mse_type'])
                    total_pose_mse += pose_mse.mean().item()
                if loss_params['mse_eff'] > 0:
                    traj_loss = out['traj_loss'].mean()
                    total_traj_loss += traj_loss.item()
                    loss = loss + traj_loss * loss_params['mse_eff']
                    # predicted traj
                    pred_traj = out['pred_traj']  # B K 4 obslen+predlen
                    gt_traj = targets['pred_traj']  # B predlen 4
                    batch_size, pred_len, n_dim = gt_traj.size()
                    obslen = pred_traj.size(3) - pred_len
                    pred_traj = pred_traj[:,:,:,obslen:].permute(0,3,1,2)  # B predlen K 4
                    try:
                        traj_mse = calc_stoch_mse(pred_traj,  # B predlen K 4
                                              gt_traj,  # B predlen 4
                                              loss_params['stoch_mse_type'])
                    except:
                        import pdb; pdb.set_trace()
                    total_traj_mse += traj_mse.mean().item()
                # ce loss
                if loss_params['cls_eff'] > 0:
                    for k in logits:
                        if n_iter == 0:
                            targets_e[k] = targets[k].detach()
                            logits_e[k] = logits[k].detach()
                        else:
                            try:
                                targets_e[k] = torch.cat((targets_e[k], targets[k].detach()), dim=0)
                            except:
                                import pdb; pdb.set_trace()
                            logits_e[k] = torch.cat((logits_e[k], logits[k].detach()), dim=0)
                    ce_dict = {}
                    for k in logits:
                        ce_dict[k] = cls_loss_func(logits[k], 
                                                   targets[k], 
                                                   weight=cls_weights_multi[k])
                        loss = loss + ce_dict[k] * loss_params['cls_eff']
                # top k mono loss
                if args.mm_fusion_mode == 'no_fusion':
                    proto_simi = [out['mm_proto_simi'][m] for m in out['mm_proto_simi']]
                    proto_simi = torch.cat(proto_simi, dim=0)  # B*M P
                else:
                    proto_simi = out['proto_simi']  # B P
                all_proto_simi.append(proto_simi.detach().cpu())
                batch_sparsity, topk_indices = calc_topk_monosem(proto_simi, 
                                                            args.topk,
                                                            args.topk_metric)
                total_batch_sparsity += batch_sparsity.mean().item()
                if loss_params['diversity_loss_eff'] > 0:
                    diversity_loss = calc_diversity_loss(model.module.proto_enc.weight)
                    total_diversity_loss += diversity_loss.item()
                    loss = loss + diversity_loss * loss_params['diversity_loss_eff']
                if loss_params['mono_sem_eff'] > 0:
                    mono_sem_loss = -batch_sparsity.mean()
                    total_mono_sem_loss += mono_sem_loss.item()
                    loss = loss + mono_sem_loss * loss_params['mono_sem_eff']
                if loss_params['mono_sem_l1_eff'] > 0:
                    mono_sem_l1_loss = batch_sparsity.abs().mean()
                    total_mono_sem_l1_loss += mono_sem_l1_loss.item()
                    loss = loss + mono_sem_l1_loss * loss_params['mono_sem_l1_eff']
                if loss_params['mono_sem_align_eff'] > 0 and loss_params['cls_eff'] > 0:
                    mono_sem_align_loss = 0
                    for act_set in logits:
                        weights = model.module.proto_dec[act_set].weight  # n_cls, n_proto
                        mono_sem_align_loss += calc_mono_sem_align_loss(weights, 
                                                                        batch_sparsity,
                                                                        loss_params['mono_sem_align_func'])
                    total_mono_sem_align_loss += mono_sem_align_loss.item()
                    loss = loss + mono_sem_align_loss * loss_params['mono_sem_align_eff']
                if loss_params['cluster_loss_eff'] > 0:
                    modality_simi_mats = calc_batch_simi_simple(feat_dict=out['enc_out'],
                                                                log_logit_scale=model.module.logit_scale,
                                                                simi_func='dot_prod',
                                                                pair_mode='pair_wise')
                    cluster_loss = calc_contrast_loss(modality_simi_mats,
                                                       pair_mode='pair_wise')
                    loss = loss + cluster_loss * loss_params['cluster_loss_eff']
                    total_cluster_loss += cluster_loss.item()
            else:
                raise NotImplementedError()
            # backward
            if is_train:
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
        if n_iter%50 == 0:
            print(f'cur mem allocated: {torch.cuda.memory_allocated(device)}')

    # calc epoch wise metric
    # all sparsity
    if args.model_name == 'pedspace':
        all_proto_simi = torch.cat(all_proto_simi, dim=0)  # n_samples(*M) P
        all_sparsity, all_topk_indices = calc_topk_monosem(all_proto_simi, 
                                                        args.topk,
                                                        args.topk_metric)
        model.module.all_sparsity = all_sparsity # P,
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
    
    # log res
    log('\n')
    # init res dict
    res = {}
    # res['cls'] = {}
    # for k in ('cross', 'atomic', 'complex', 'communicative', 'transporting', 'age'):
    #     res['cls'][k] = {
    #         'acc': 0,
    #         'f1': 0,
    #         'map': 0,
    #     }
    # res['cls']['cross']['auc'] = 0
    # update res dict
    if loss_params['cls_eff'] > 0:
        res['cls'] = {}
        for k in acc_e:
            if k == 'cross':
                res['cls'][k] = {
                    'acc': acc_e[k],
                    'map': mAP_e[k],
                    'f1': f1_e[k],
                    'auc': auc_cross,
                    # 'logits': logits_e['cross'].detach().cpu().numpy(),
                }
                log(f'\tacc: {acc_e[k]}\t mAP: {mAP_e[k]}\t f1: {f1_e[k]}\t f1b: {f1b_e[k]}\t AUC: {auc_cross}')
                log(f'\tprecision: {prec_e[k]}')
                log(f'\tconf mat: {conf_mat}')
                log(f'\tconf mat norm: {conf_mat_norm}')
            else:
                res['cls'][k] = {
                    'acc': acc_e[k],
                    'f1': f1_e[k],
                    'map': mAP_e[k],
                    'auc': 0,
                    # 'logits': logits_e[k]
                }
                log(f'\t{k} acc: {acc_e[k]}\t {k} mAP: {mAP_e[k]}\t {k} f1: {f1_e[k]}')
                log(f'\t{k} recall: {rec_e[k]}')
                log(f'\t{k} precision: {prec_e[k]}')
    

    if loss_params['mse_eff'] > 0:
        res['traj_mse'] = total_traj_mse / (n_iter+1)
        log(f'\t traj mse: {total_traj_mse / (n_iter+1)}')
    if loss_params['pose_mse_eff'] > 0:
        res['pose_mse'] = total_pose_mse / (n_iter+1)
        log(f'\t pose mse: {total_pose_mse / (n_iter+1)}')
    if loss_params['logsig_loss_eff'] > 0:
        res['logsig_loss'] = total_logsig_loss / (n_iter+1)
        log(f'\t logsig loss: {total_logsig_loss / (n_iter+1)}')
    if loss_params['diversity_loss_eff'] > 0:
        res['diversity_loss'] = total_diversity_loss / (n_iter+1)
        log(f'\t diversity loss: {total_diversity_loss / (n_iter+1)}')
    if loss_params['mono_sem_eff'] > 0:
        res['mono_sem_loss'] = total_mono_sem_loss / (n_iter+1)
        log(f'\t mono_sem_loss: {total_mono_sem_loss / (n_iter+1)}')
    if loss_params['mono_sem_l1_eff'] > 0:
        res['mono_sem_l1_loss'] = total_mono_sem_l1_loss / (n_iter+1)
        log(f'\t mono_sem_l1_loss: {total_mono_sem_l1_loss / (n_iter+1)}')
    if loss_params['mono_sem_align_eff'] > 0:
        res['mono_sem_align_loss'] = total_mono_sem_align_loss / (n_iter+1)
        log(f'\t mono_sem_align_loss: {total_mono_sem_align_loss / (n_iter+1)}')
    if loss_params['cluster_loss_eff'] > 0:
        res['cluster_loss'] = total_cluster_loss / (n_iter+1)
        log(f'\t cluster_loss: {total_cluster_loss / (n_iter+1)}')
    if model_name == 'pedspace':
        res['batch_sparsity'] = total_batch_sparsity / (n_iter+1)
        log(f'\t mean batch sparsity: {total_batch_sparsity / (n_iter+1)}')
        res['all_sparsity'] = all_sparsity.mean().item()
        log(f'\t all sparsity: {all_sparsity.mean().item()}')
    log('\n')
    return res
