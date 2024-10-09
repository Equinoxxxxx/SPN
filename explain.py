import torch
import numpy as np
from tqdm import tqdm
import os
import cv2
import copy
import argparse
import os
import pickle

from tools.utils import makedir, write_info_txt
from tools.datasets.identify_sample import get_ori_img_path, get_sklt_img_path
from tools.datasets.dataset_id import ID_TO_DATASET
from tools.data.normalize import recover_norm_imgs, img_mean_std_BGR, recover_norm_sklt, recover_norm_bbox
from tools.visualize.heatmap import visualize_featmap3d
from tools.visualize.visualize_skeleton import visualize_sklt_with_pseudo_heatmap
from tools.visualize.visualize_bbox import draw_boxes_on_img
from tools.visualize.visualize_1d_seq import vis_1d_seq
from tools.visualize.visualize_neighbor_bbox import visualize_neighbor_bbox
from tools.data.resize_img import resize_image


def forwad_pass(dataloader,
                model_parallel,
                device='cuda:0',
                modalities=None,
                ):
    model_parallel.module.eval()
    all_inputs = []
    all_targets = []
    all_info = []
    all_outputs = []
    all_batch_size = []
    with torch.no_grad():
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
                inputs['traj_unnormed'] = data['obs_bboxes_unnormed'].to(device)  # B T 4
            if 'ego' in modalities:
                inputs['ego'] = data['obs_ego'].to(device)  # B T 1
            if 'social' in modalities:
                inputs['social'] = data['obs_neighbor_relation'].to(device)
                inputs['obs_neighbor_bbox'] = data['obs_neighbor_bbox'].to(device)
                inputs['obs_neighbor_oid'] = data['obs_neighbor_oid'].to(device)
            # load gt
            targets = {}
            targets['cross'] = data['pred_act'].to(device).view(-1) # idx, not one hot
            targets['atomic'] = data['atomic_actions'].to(device).view(-1)
            targets['complex'] = data['complex_context'].to(device).view(-1)
            targets['communicative'] = data['communicative'].to(device).view(-1)
            targets['transporting'] = data['transporting'].to(device).view(-1)
            targets['age'] = data['age'].to(device).view(-1)
            targets['pred_traj'] = data['pred_bboxes'].to(device)  # B T 4
            targets['pred_sklt'] = data['pred_skeletons'].to(device)  # B ndim predlen nj
            # other info
            info = {}
            for k in data.keys():
                if k not in ['ped_imgs', 'obs_skeletons', 'obs_context', 'obs_bboxes', 'obs_ego', 'obs_neighbor_relation',
                             'pred_act', 'atomic_actions', 'complex_context', 'communicative', 'transporting', 'age',
                             'pred_bboxes', 'pred_skeletons']:
                    info[k] = data[k]
            # forward
            _out = model_parallel((inputs, targets), is_train=0)
            # save
            for k in inputs.keys():
                inputs[k] = inputs[k].detach().cpu()
            for k in targets.keys():
                targets[k] = targets[k].detach().cpu()
            for k in info.keys():
                info[k] = info[k].detach().cpu()
                # print(k, info[k].shape)
            out = {}
            for k in _out.keys():
                if k == 'proto_simi':
                    out[k] = _out[k].detach().cpu()
                elif k in ('feat', 'modality_effs'):
                    out[k] = {k2:_out[k][k2].detach().cpu() for k2 in _out[k].keys()}
            all_inputs.append(inputs)
            all_targets.append(targets)
            all_info.append(info)  # info: B ...
            all_outputs.append(out)
            all_batch_size.append(inputs[list(inputs.keys())[0]].shape[0])
            if n_iter%50 == 0:
                print(f'cur mem allocated: {torch.cuda.memory_allocated(device)}')
    return all_inputs, all_targets, all_info, all_outputs, all_batch_size


def select_topk(dataloader,
                model_parallel,
                args,
                device='cuda:0',
                modalities=None,
                save_root=None,
                log=print):
    log(f'Explain top samples')
    log(f'Getting forward pass results')
    all_inputs, all_targets, all_info, all_outputs, all_batch_size = forwad_pass(dataloader,
                                                                                model_parallel,
                                                                                device,
                                                                                modalities)
    nm_to_dataset = {}
    for d in dataloader.dataset.datasets:
        nm_to_dataset[d.dataset_name] = d
    batch_size = all_batch_size[0]
    n_samples = sum(all_batch_size)
    all_proto_simi = torch.cat([out['proto_simi'] for out in all_outputs], dim=0)  # n_samples P
    simi_mean = all_proto_simi.mean(dim=0)  # (P)
    simi_var = all_proto_simi.var(dim=0, unbiased=True)  # (P)
    all_relative_var = (all_proto_simi - simi_mean.unsqueeze(0))**2 / (simi_var.unsqueeze(0) + 1e-5)  # (n_samples, P)
    top_k_relative_var, top_k_rel_var_indices = torch.topk(all_relative_var, args.topk_explain, dim=0)  # (k, P)
    if args.topk_metric_explain == 'activation':
        top_k_values, top_k_indices = torch.topk(all_proto_simi, args.topk_explain, dim=0)  # (k, P) (k, P)
    elif args.topk_metric_explain == 'relative_var':
        top_k_values, top_k_indices = top_k_relative_var, top_k_rel_var_indices
    else:
        raise ValueError(args.topk_metric_explain)
    
    K,P = top_k_indices.shape
    # get info
    info_cat = {k:[] for k in all_info[0].keys()}
    for info in all_info:
        for k in info.keys():
            info_cat[k].append(info[k])
    for k in info_cat.keys():
        info_cat[k] = torch.cat(info_cat[k], dim=0)  # n_samples, ...

    # visualize and save
    # sample ids
    all_sample_ids = {'dataset_name':[],
                  'set_id_int':[],
                  'vid_id_int':[],
                  'img_nm_int':[],
                  'ped_id_int':[],}
    for k in all_sample_ids.keys():
        all_sample_ids[k] = info_cat[k]
        # for info in info_cat[k]:
        #     all_sample_ids[k].append(info)
        # all_sample_ids[k] = torch.stack(all_sample_ids[k], dim=0)  # n_samples,...
    # modality weights
    all_modality_effs = None
    if all_outputs[0]['modality_effs']:
        all_modality_effs = {k:[] for k in all_outputs[0]['modality_effs'].keys()}
        for out in all_outputs:
            for k in out['modality_effs'].keys():
                all_modality_effs[k].append(out['modality_effs'][k])  # B
        for k in all_modality_effs.keys():
            all_modality_effs[k] = torch.cat(all_modality_effs[k], dim=0)  # n_samples,
    # act cls and last linear weights
    all_act_cls = {k:[] for k in all_targets[0].keys()}
    all_act_cls.pop('pred_traj')
    all_act_cls.pop('pred_sklt')
    for target in all_targets:
        for k in all_act_cls.keys():
            all_act_cls[k].append(copy.deepcopy(target[k]))
    for k in all_act_cls.keys():
        all_act_cls[k] = torch.cat(all_act_cls[k], dim=0)  # n_samples,
    # other info
    log(f'Saving sample info')
    explain_info = []
    tbar = tqdm(range(P), miniters=1)
    for p in tbar:
        last_weights_cur_proto = {act:model_parallel.module.proto_dec[act].weight[:,p].detach().cpu().numpy() \
                                  for act in model_parallel.module.proto_dec.keys()}
        cur_p_rel_var = top_k_relative_var[:,p].mean().cpu().numpy()
        explain_info.append({'mean_rel_var':cur_p_rel_var,
                                'last_weights':last_weights_cur_proto,
                                'sample_info':[],
                                'proto_id':p})
        for k in range(K):
            sample_idx = top_k_indices[k,p]
            proto_simi = copy.deepcopy(all_proto_simi[sample_idx].detach().cpu().numpy()) # P
            sample_ids = {k:all_sample_ids[k][sample_idx] for k in all_sample_ids.keys()}
            modality_effs = None
            if all_modality_effs is not None:
                modality_effs = {k:copy.deepcopy(all_modality_effs[k][sample_idx].detach().cpu().numpy()) \
                                 for k in all_modality_effs.keys()}
            act_cls = {act:all_act_cls[act][sample_idx].detach().cpu().numpy() for act in all_act_cls.keys()}
            
            content = [f'mean relative var of cur proto: {cur_p_rel_var}', 
                       f'relative var of cur sample: {top_k_relative_var[k,p].item()}', 
                       f'sample ids: {sample_ids}\n', 
                       f'modality effs: {modality_effs}\n',
                       f'labels: {act_cls}\n', 
                       f'proto_simi: {proto_simi[p]}\n',
                       f'last weights of cur proto: {last_weights_cur_proto}\n',
                       ]
            save_path = os.path.join(save_root, str(p), str(k))
            makedir(save_path)
            write_info_txt(content, 
                           os.path.join(save_path, 'sample_info.txt'))
            explain_info[p]['sample_info'].append({'modality_effs': modality_effs,
                                                              'labels': act_cls,
                                                              'images': {m:None for m in modality_effs},
                                                              'rel_var': top_k_relative_var[k,p].cpu().numpy(),
                                                                'proto_simi': proto_simi[p],
                                                            })
    
    mm_res = {}
    # img
    if 'img' in modalities:
        log(f'Saving img explanation')
        mm_res['img'] = []
        all_img = torch.cat([inp['img'] for inp in all_inputs], dim=0)  # n_samples 3 T H W
        selected_imgs = copy.deepcopy(torch.gather(all_img, 0, top_k_indices.view(-1,1,1,1,1).\
                                expand(K*P, 3, all_img.shape[2], all_img.shape[3], all_img.shape[4])))  # K*P 3 T H W
        # recover from normalization
        selected_imgs = selected_imgs.permute(1,2,3,4,0)  # 3 T H W K*P
        if args.model_color_order == 'RGB':
            selected_imgs = selected_imgs[[2,1,0],:,:,:,:]
        img_mean, img_std = img_mean_std_BGR(args.img_norm_mode)  # BGR
        selected_imgs = recover_norm_imgs(selected_imgs, img_mean, img_std)  # 3 T H W K*P
        selected_imgs = selected_imgs.permute(4,1,2,3,0).reshape(K,P,all_img.shape[2],all_img.shape[3],all_img.shape[4], 3)  # K P T H W 3
        # 2D case
        if 'deeplab' in args.img_backbone_name or 'vit' in args.img_backbone_name:
            selected_imgs = selected_imgs[:,:,-1:,:,:,:]  # K P 1 H W 3
        _,_,T,H,W,_ = selected_imgs.shape
        # get feature map
        all_feat = torch.cat([out['feat']['img'] for out in all_outputs], dim=0)  # n_samples C (T) H W
        if len(all_feat.shape) == 4:
            all_feat = all_feat.unsqueeze(2)  # n_samples C 1 H W
        selected_feat = torch.gather(all_feat, 0, top_k_indices.view(-1,1,1,1,1).\
                                expand(K*P, all_feat.shape[1], all_feat.shape[2], all_feat.shape[3], all_feat.shape[4]))  # K*P C T/1 H W
        selected_feat = selected_feat.permute(0,2,3,4,1).\
            reshape(K,P,all_feat.shape[2], all_feat.shape[3], all_feat.shape[4],all_feat.shape[1])  # K P T/1 H W C
        # visualize and save
        for p in tqdm(range(P)):
            mm_res['img'].append([])
            for k in range(K):
                img = selected_imgs[k,p].cpu().int().numpy()  # T/1 H W 3
                feat = selected_feat[k,p].cpu().numpy()  # T/1 H W C
                save_path = os.path.join(save_root, str(p), str(k), 'img')
                makedir(save_path)
                mean_dir = os.path.join(save_path, 'mean')
                makedir(mean_dir)
                max_dir = os.path.join(save_path, 'max')
                makedir(max_dir)
                min_dir = os.path.join(save_path, 'min')
                makedir(min_dir)
                mean_mean, mean_max, mean_min, mean_overlay_imgs, heatmaps = visualize_featmap3d(feat,img, mode='mean', save_dir=mean_dir)
                max_mean, max_max, max_min, _, _ = visualize_featmap3d(feat,img, mode='max', save_dir=max_dir)
                min_mean, min_max, min_min, _, _ = visualize_featmap3d(feat,img, mode='min', save_dir=min_dir)
                # write_info_txt([mean_mean, mean_max, mean_min, max_mean, max_max, max_min, min_mean, min_max, min_min],
                #                os.path.join(save_path, 'feat_info.txt'))
                max_t = np.argmax(np.max(heatmaps, axis=(1,2,3)))
                mm_res['img'][p].append(mean_overlay_imgs[max_t])
                explain_info[p]['sample_info'][k]['images']['img'] = mean_overlay_imgs[max_t]
    # ctx
    if 'ctx' in modalities:
        log(f'Saving ctx explanation')
        mm_res['ctx'] = []
        all_img = torch.cat([inp['ctx'] for inp in all_inputs], dim=0)  # n_samples 3 T H W
        selected_imgs = copy.deepcopy(torch.gather(all_img, 
                                                   0, 
                                                   top_k_indices.view(-1,1,1,1,1).expand(K*P, 3, all_img.shape[2], all_img.shape[3], all_img.shape[4])))  # K*P 3 T H W
        # recover from normalization
        selected_imgs = selected_imgs.permute(1,2,3,4,0)  # 3 T H W K*P
        if args.model_color_order == 'RGB':
            selected_imgs = selected_imgs[[2,1,0],:,:,:,:]
        img_mean, img_std = img_mean_std_BGR(args.img_norm_mode)  # BGR
        selected_imgs = recover_norm_imgs(selected_imgs, img_mean, img_std)  # 3 T H W K*P
        selected_imgs = selected_imgs.permute(4,1,2,3,0).reshape(K,P,all_img.shape[2],all_img.shape[3],all_img.shape[4], 3)  # K P T H W 3
        # 2D case
        if 'deeplab' in args.img_backbone_name or 'vit' in args.img_backbone_name:
            selected_imgs = selected_imgs[:,:,-1:,:,:,:]  # K P 1 H W 3
        _,_,T,H,W,_ = selected_imgs.shape
        # get feature map
        all_feat = torch.cat([out['feat']['ctx'] for out in all_outputs], dim=0)  # n_samples C (T) H W
        if len(all_feat.shape) == 4:
            all_feat = all_feat.unsqueeze(2)  # n_samples C 1 H W
        selected_feat = copy.deepcopy(torch.gather(all_feat, 0, top_k_indices.view(-1,1,1,1,1).\
                                expand(K*P, all_feat.shape[1], all_feat.shape[2], all_feat.shape[3], all_feat.shape[4])))  # K*P C T/1 H W
        selected_feat = selected_feat.permute(0,2,3,4,1).\
            reshape(K,P,all_feat.shape[2], all_feat.shape[3], all_feat.shape[4],all_feat.shape[1])  # K P T/1 H W C
        # visualize and save
        for p in tqdm(range(P)):
            mm_res['ctx'].append([])
            for k in range(K):
                img = selected_imgs[k,p].cpu().int().numpy()  # T/1 H W 3
                feat = selected_feat[k,p].cpu().numpy()  # T/1 H W C
                save_path = os.path.join(save_root, str(p), str(k), 'ctx')
                makedir(save_path)
                mean_dir = os.path.join(save_path, 'mean')
                makedir(mean_dir)
                max_dir = os.path.join(save_path, 'max')
                makedir(max_dir)
                min_dir = os.path.join(save_path, 'min')
                makedir(min_dir)
                mean_mean, mean_max, mean_min, mean_overlay_imgs, heatmaps = visualize_featmap3d(feat,img, mode='mean', save_dir=mean_dir)
                max_mean, max_max, max_min, _, _ = visualize_featmap3d(feat,img, mode='max', save_dir=max_dir)
                min_mean, min_max, min_min, _, _ = visualize_featmap3d(feat,img, mode='min', save_dir=min_dir)
                # write_info_txt([mean_mean, mean_max, mean_min, max_mean, max_max, max_min, min_mean, min_max, min_min],
                #                os.path.join(save_path, 'feat_info.txt'))
                max_t = np.argmax(np.max(heatmaps, axis=(1,2,3)))
                mm_res['ctx'][p].append(mean_overlay_imgs[max_t])
                explain_info[p]['sample_info'][k]['images']['ctx'] = mean_overlay_imgs[max_t]
    # sklt TBD
    if 'sklt' in modalities:
        log(f'Saving sklt explanation')
        mm_res['sklt'] = []
        if 'coord' in args.sklt_format and 'transformer' in args.sklt_backbone_name:
            all_traj = torch.cat([inp['traj_unnormed'] for inp in all_inputs], dim=0)  # n_samples T 4
            all_sklt = torch.cat([inp['sklt'] for inp in all_inputs], dim=0)  # n_samples 2 T nj
            _, nd, obslen, nj = all_sklt.shape
            all_feat = torch.cat([out['feat']['sklt'] for out in all_outputs], dim=0)  # n_samples T
            # visualize and save
            for p in tqdm(range(P)):
                mm_res['sklt'].append([])
                for k in range(K):
                    save_path = os.path.join(save_root, str(p), str(k), 'sklt')
                    makedir(save_path)
                    sam_idx = top_k_indices[k,p]
                    set_id = all_sample_ids['set_id_int'][sam_idx]
                    vid_id = all_sample_ids['vid_id_int'][sam_idx]
                    img_nms = all_sample_ids['img_nm_int'][sam_idx]
                    obj_id = all_sample_ids['ped_id_int'][sam_idx]
                    dataset_name = all_sample_ids['dataset_name'][sam_idx].detach().item()
                    dataset_name = ID_TO_DATASET[dataset_name]  # int --> str
                    sklt_coords = copy.deepcopy(all_sklt[sam_idx,:].detach().cpu())  # 2 T nj
                    # de-normalize
                    if '0-1' in args.sklt_format:
                        sklt_coords = recover_norm_sklt(sklt_coords, dataset_name)  # 2 T nj (int)
                    sklt_coords = sklt_coords.permute(1,2,0) # T nj 2
                    traj = copy.deepcopy(all_traj[sam_idx, :].detach().cpu().numpy())  # T 4
                    # de-normalize
                    if '0-1' in args.traj_format:
                        traj = recover_norm_bbox(traj, dataset_name)  # T 4 (int)
                    traj = traj.astype(np.int32)
                    feat = copy.deepcopy(all_feat[sam_idx].detach().cpu().numpy())  # obslen*nj
                    feat = feat.reshape(obslen, nj)  # obslen nj
                    sklt_imgs = []
                    for img_nm in img_nms:
                        img_path = get_sklt_img_path(dataset_name,
                                                    set_id=set_id,
                                                    vid_id=vid_id,
                                                    obj_id=obj_id,
                                                    img_nm=img_nm,
                                                    with_sklt=True,
                                                    )
                        sklt_imgs.append(cv2.imread(img_path))
                    sklt_imgs = np.stack(sklt_imgs, axis=0)  # obslen h w 3
                    # visualize
                    overlay_imgs, heatmaps = visualize_sklt_with_pseudo_heatmap(sklt_imgs, 
                                                                                sklt_coords, 
                                                                                feat, 
                                                                                traj, 
                                                                                dataset_name, 
                                                                                save_path)
                    max_t = np.argmax(np.max(heatmaps, axis=(1,2,3)))
                    mm_res['sklt'][p].append(overlay_imgs[max_t])
                    explain_info[p]['sample_info'][k]['images']['sklt'] = overlay_imgs[max_t]
    if 'traj' in modalities:
        log(f'Saving traj explanation')
        mm_res['traj'] = []
        all_traj = torch.cat([inp['traj_unnormed'] for inp in all_inputs], dim=0)  # n_samples T 4
        all_feat = None
        if 'transformer' in args.traj_backbone_name:
            all_feat = torch.cat([out['feat']['traj'] for out in all_outputs], dim=0) # n_samples T
        for p in tqdm(range(P)):
            mm_res['traj'].append([])
            for k in range(K):
                save_path = os.path.join(save_root, str(p), str(k), 'traj')
                makedir(save_path)
                sam_idx = top_k_indices[k,p]
                set_id = all_sample_ids['set_id_int'][sam_idx]
                vid_id = all_sample_ids['vid_id_int'][sam_idx]
                img_nms = all_sample_ids['img_nm_int'][sam_idx]
                obj_id = all_sample_ids['ped_id_int'][sam_idx]
                dataset_name = all_sample_ids['dataset_name'][sam_idx].detach().item()
                dataset_name = ID_TO_DATASET[dataset_name]  # int --> str
                traj = copy.deepcopy(all_traj[sam_idx].detach().cpu().numpy())
                # de-normalize
                if '0-1' in args.traj_format:
                    traj = recover_norm_bbox(traj, dataset_name)  # T 4 (int)
                traj = traj.astype(np.int32)
                bg_img_path = get_ori_img_path(nm_to_dataset[dataset_name],
                                            set_id=set_id,
                                            vid_id=vid_id,
                                            img_nm=img_nms[-1],
                                            )
                bg_img = cv2.imread(bg_img_path)
                bg_blank = np.ones_like(bg_img)*220
                if 'transformer' in args.traj_backbone_name:
                    feat = copy.deepcopy(all_feat[sam_idx].cpu())
                traj_img = draw_boxes_on_img(bg_img, traj)
                blank_traj_img = draw_boxes_on_img(bg_blank, traj)
                cv2.imwrite(filename=os.path.join(save_path, 'traj.png'), 
                            img=traj_img)
                cv2.imwrite(filename=os.path.join(save_path, 'traj_blank_bg.png'), 
                            img=blank_traj_img)
                mm_res['traj'][p].append(blank_traj_img)
                explain_info[p]['sample_info'][k]['images']['traj'] = blank_traj_img

    if 'ego' in modalities:
        log(f'Saving ego explanation')
        mm_res['ego'] = []
        all_ego = torch.cat([inp['ego'] for inp in all_inputs], dim=0) # n_samples T 1
        max_ego = all_ego.max().item()
        min_ego = all_ego.min().item()
        lim = (min_ego, max_ego)
        all_feat = torch.cat([out['feat']['ego'] for out in all_outputs], dim=0)  # n_samples T
        for p in tqdm(range(P)):
            mm_res['ego'].append([])
            for k in range(K):
                save_path = os.path.join(save_root, str(p), str(k), 'ego')
                sam_idx = top_k_indices[k,p]
                ego = copy.deepcopy(all_ego[sam_idx,:,0].cpu().detach().numpy()) # T
                feat = copy.deepcopy(all_feat[sam_idx].cpu().detach().numpy())  # T
                ego_img = vis_1d_seq(ego, lim, save_path, weights=None)
                mm_res['ego'][p].append(ego_img)
                explain_info[p]['sample_info'][k]['images']['ego'] = ego_img

    if 'social' in modalities:
        log(f'Saving social explanation')
        mm_res['social'] = []
        all_feat = None
        if 'transformer' in args.social_backbone_name:
            all_feat = torch.cat([out['feat']['social'] for out in all_outputs], dim=0) # n_samples T
            all_traj = torch.cat([inp['traj_unnormed'] for inp in all_inputs], dim=0)
            all_neighbor_bbox = torch.cat([inp['obs_neighbor_bbox'] for inp in all_inputs], dim=0) # n_samples K T 4
            for p in tqdm(range(P)):
                mm_res['social'].append([])
                for k in range(K):
                    save_path = os.path.join(save_root, str(p), str(k), 'social')
                    makedir(save_path)
                    sam_idx = top_k_indices[k,p]
                    set_id = all_sample_ids['set_id_int'][sam_idx].cpu()
                    vid_id = all_sample_ids['vid_id_int'][sam_idx].cpu()
                    img_nms = all_sample_ids['img_nm_int'][sam_idx].cpu()
                    obj_id = all_sample_ids['ped_id_int'][sam_idx].cpu()
                    dataset_name = all_sample_ids['dataset_name'][sam_idx].item()
                    dataset_name = ID_TO_DATASET[dataset_name]  # int --> str
                    traj = copy.deepcopy(all_traj[sam_idx, :].detach().cpu().numpy())
                    # # de-normalize
                    if '0-1' in args.traj_format:
                        traj = recover_norm_bbox(traj, dataset_name)  # T 4 (int)
                    traj = traj.astype(np.int32)
                    neighbor_bbox = copy.deepcopy(all_neighbor_bbox[sam_idx].cpu().detach().numpy())
                    neighbor_bbox = neighbor_bbox.astype(np.int32)  # K T 4
                    n_neighbor, obslen, _ = neighbor_bbox.shape
                    weights = copy.deepcopy(all_feat[sam_idx].cpu().detach().numpy())  # obslen*n_neighbor
                    weights = weights.reshape(obslen, n_neighbor)  # obslen n_neighbor
                    bg_img_path = get_ori_img_path(nm_to_dataset[dataset_name],
                                                set_id=set_id,
                                                vid_id=vid_id,
                                                img_nm=img_nms[-1],
                                                )
                    bg_img = cv2.imread(bg_img_path)
                    try:
                        social_img = visualize_neighbor_bbox(bg_img,traj[-1],neighbor_bbox[:,-1],weights=weights[-1])
                    except:
                        import pdb; pdb.set_trace()
                    cv2.imwrite(filename=os.path.join(save_path, 'social.png'), 
                                img=social_img)
                    mm_res['social'][p].append(social_img)
                    explain_info[p]['sample_info'][k]['images']['social'] = social_img
    # TBD: all samples from one modality together
    # mm_res: {modality: P*[K*array(H W 3)]}
    log(f'Plotting all explanation in {save_root}')
    plot_all_explanation(explain_info, os.path.join(save_root, 'all_explanation.png'))


def plot_all_explanation(explain_info, path, part_height=250, part_width=250, spacing=50):
    '''
    explain_info: P*[{'mean_rel_var':float, 
                      'last_weights':{act:array}, 
                        'proto_id':int,
                      'sample_info':[{'rel_var':float,
                                      'labels':{act:int}, 
                                      'images':{modality:array}, 
                                      'modality_effs':{modality:float}, 
                                      'proto_simi':float,
                                      }]
                      }]
    path: str, the path to save the plot
    row_spacing: float, the spacing between rows
    col_spacing: float, the spacing between columns
    '''

    P = len(explain_info)
    K = len(explain_info[0]['sample_info'])
    M = len(explain_info[0]['sample_info'][0]['images'])

    # rank by mean_rel_var
    explain_info = sorted(explain_info, key=lambda x: x['mean_rel_var'], reverse=True)

    # Calculate the total height and width of the canvas
    total_height = P * K * (part_height + spacing)
    total_width = (M+3) * (part_width + spacing)

    # Create a blank canvas
    canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

    for i, block in enumerate(explain_info):
        block_y_offset = i * K * (part_height + spacing)
        
        # Draw mean_rel_var
        mean_rel_var = block['mean_rel_var']
        proto_title = f'prototype {block["proto_id"]}'
        mean_rel_var_text = f"mean_rel_var: {round(float(mean_rel_var),3)}"
        cv2.putText(canvas, proto_title, (spacing, block_y_offset + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, mean_rel_var_text, (spacing, block_y_offset + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Draw sample info
        for j, sample in enumerate(block['sample_info']):
            sample_y_offset = block_y_offset + j * (part_height + spacing)
            sample_left_text = [f'sample {j}', 
                                f"rel_var: {sample['rel_var']}", 
                                '    labels:'
                                ]
            for act in sample['labels']:
                sample_left_text.append(f"    {act}: {sample['labels'][act]}")
            for l in range(len(sample_left_text)):
                cv2.putText(canvas, sample_left_text[l], (spacing+50, sample_y_offset + part_height//4 + l*15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            for k, (modality, image) in enumerate(sample['images'].items()):
                img_x_offset = spacing + (k + 1) * (part_width + spacing)
                img_y_offset = sample_y_offset
                img_resized = resize_image(image, (part_width, part_height), mode='pad', padding_color=(255, 255, 255))
                canvas[img_y_offset:img_y_offset + part_height, img_x_offset:img_x_offset + part_width] = img_resized
                modality_eff_text = f"modality eff: {round(float(sample['modality_effs'][modality]),3)}"
                cv2.putText(canvas, modality, (img_x_offset+spacing, img_y_offset + part_height + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(canvas, modality_eff_text, (img_x_offset+spacing, img_y_offset + part_height + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            proto_simi_text = f"proto_simi: {round(float(sample['proto_simi']),3)}"
            cv2.putText(canvas, proto_simi_text, (spacing + (M + 1) * (part_width + spacing), sample_y_offset + part_height // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Draw last_weights
        weights_y_offset = block_y_offset + spacing
        for act, cls in block['last_weights'].items():
            weights_text = f"{act}:"
            cv2.putText(canvas, weights_text, ((M+2)*(part_width+spacing), weights_y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            weights_cls_y_offset = weights_y_offset + 20
            for c in range(len(cls)):
                weights_text = f"  {c}: {round(float(cls[c]),3)}"
                cv2.putText(canvas, weights_text, ((M+2)*(part_width+spacing), weights_cls_y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                weights_cls_y_offset += 20
            weights_y_offset += spacing
    
    # Save the final image
    cv2.imwrite(path, canvas)

    return canvas


