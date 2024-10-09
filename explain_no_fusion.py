import torch
import numpy as np
from tqdm import tqdm
import os
import cv2
import copy

from tools.utils import makedir, write_info_txt
from tools.datasets.identify_sample import get_ori_img_path, get_sklt_img_path,\
    DATASET_TO_ID, ID_TO_DATASET, MODALITY_TO_ID, ID_TO_MODALITY, \
        LABEL_TO_CROSSING, LABEL_TO_ATOMIC_CHOSEN, LABEL_TO_SIMPLE_CONTEXTUAL, LABEL_TO_COMPLEX_CONTEXTUAL, \
            LABEL_TO_COMMUNICATIVE, LABEL_TO_TRANSPORTIVE, LABEL_TO_AGE
from tools.data.normalize import recover_norm_imgs, img_mean_std_BGR, recover_norm_sklt, recover_norm_bbox
from tools.visualize.heatmap import visualize_featmap3d
from tools.visualize.visualize_skeleton import visualize_sklt_with_pseudo_heatmap
from tools.visualize.visualize_bbox import draw_boxes_on_img
from tools.visualize.visualize_1d_seq import vis_1d_seq
from tools.visualize.visualize_neighbor_bbox import visualize_neighbor_bbox
from tools.data.resize_img import resize_image


def forwad_pass_no_fusion(dataloader,
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
                inputs['traj_ori'] = data['obs_bboxes_ori'].to(device)  # B T 4
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
            targets['simple'] = data['simple_context'].to(device).view(-1)
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
                elif k in ('feat', 'mm_proto_simi'):
                    out[k] = {k2:_out[k][k2].detach().cpu() for k2 in _out[k].keys()}
            all_inputs.append(inputs)  # img sklt ctx traj traj_unnormed traj_ori ego social obs_neighbor_bbox
            all_targets.append(targets)  # cross atomic complex communicative transporting age pred_traj pred_sklt
            all_info.append(info)  # set_id vid_id obj_id img_nm
            all_outputs.append(out)
            # all_batch_size.append(inputs[list(inputs.keys())[0]].shape[0])
            if n_iter%50 == 0:
                print(f'cur mem allocated: {torch.cuda.memory_allocated(device)}')
    return all_inputs, all_targets, all_info, all_outputs


def select_topk_no_fusion(args,
                            dataloader,
                            model_parallel,
                            device='cuda:0',
                            modalities=None,
                            save_root=None,
                            log=print,
                            ):
    log(f'Explain top samples')
    log(f'Getting forward pass results')
    all_inputs_batches, all_targets_batches, all_info_batches, all_outputs_batches = forwad_pass_no_fusion(dataloader,
                                                                            model_parallel,
                                                                            device,
                                                                            modalities)
    nm_to_dataset = {}
    for d in dataloader.dataset.datasets:
        nm_to_dataset[d.dataset_name] = d
    
    # concat all data
    M = len(modalities)
    # inputs
    all_inputs = {}
    for m in all_inputs_batches[0].keys():
        all_inputs[m] = torch.cat([inp[m] for inp in all_inputs_batches], dim=0)  # n_samples
    N = all_inputs[list(all_inputs.keys())[0]].shape[0]
    # act cls
    all_act_cls = {k:[] for k in all_targets_batches[0].keys()}
    all_act_cls.pop('pred_traj')
    all_act_cls.pop('pred_sklt')
    for target in all_targets_batches:
        for k in all_act_cls.keys():
            all_act_cls[k].append(copy.deepcopy(target[k]))
    for k in all_act_cls.keys():
        all_act_cls[k] = torch.cat(all_act_cls[k], dim=0)  # n_samples,
    # info
    info_cat = {k:[] for k in all_info_batches[0].keys()}
    for info in all_info_batches:
        for k in info.keys():
            info_cat[k].append(info[k])
    for k in info_cat.keys():
        info_cat[k] = torch.cat(info_cat[k], dim=0)  # n_samples, ...
    all_sample_ids = {'dataset_name':[],
                  'set_id_int':[],
                  'vid_id_int':[],
                  'img_nm_int':[],
                  'ped_id_int':[],}
    for k in all_sample_ids.keys():
        all_sample_ids[k] = info_cat[k]
    all_modality_ids_stack = torch.cat([MODALITY_TO_ID[m]*torch.ones(N, dtype=torch.long) for m in modalities], dim=0)  # n_samples*M
    # outputs
    mm_proto_simi = {k:[] for k in all_outputs_batches[0]['mm_proto_simi'].keys()}
    for out in all_outputs_batches:
        for k in mm_proto_simi.keys():
            mm_proto_simi[k].append(out['mm_proto_simi'][k])
    for k in mm_proto_simi.keys():
        mm_proto_simi[k] = torch.cat(mm_proto_simi[k], dim=0)
    mm_proto_simi_stack = torch.cat([mm_proto_simi[k] for k in mm_proto_simi.keys()], dim=0)  # n_samples*M, P
    all_feat = {k:[] for k in all_outputs_batches[0]['feat'].keys()}
    for out in all_outputs_batches:
        for k in all_feat.keys():
            all_feat[k].append(out['feat'][k])
    for k in all_feat.keys():
        all_feat[k] = torch.cat(all_feat[k], dim=0) # n_samples ...

    # select topk samples
    simi_mean = mm_proto_simi_stack.mean(dim=0)  # (P)
    simi_var = mm_proto_simi_stack.var(dim=0, unbiased=True)  # (P)
    all_relative_var = (mm_proto_simi_stack - simi_mean.unsqueeze(0))**2 / (simi_var.unsqueeze(0) + 1e-5)  # (n_samples, P)
    top_k_relative_var, top_k_rel_var_indices = torch.topk(all_relative_var, args.topk_explain, dim=0)  # (k, P)
    if args.topk_metric_explain == 'activation':
        top_k_values, top_k_indices = torch.topk(mm_proto_simi_stack, args.topk_explain, dim=0)  # (k, P) (k, P)
    elif args.topk_metric_explain == 'relative_var':
        top_k_values, top_k_indices = top_k_relative_var, top_k_rel_var_indices
    else:
        raise ValueError(args.topk_metric_explain)
    
    K,P = top_k_indices.shape
    # save
    log(f'Saving sample info')
    explain_info = []
    tbar = tqdm(range(P), miniters=1)
    for p in tbar:
        last_weights_cur_proto = {act:model_parallel.module.proto_dec[act].weight[:,p].detach().cpu().numpy() \
                                  for act in model_parallel.module.proto_dec.keys()}
        cur_p_rel_var = top_k_relative_var[:,p].mean().cpu().numpy()
        explain_info.append({'mean_rel_var':cur_p_rel_var,
                            'last_weights':last_weights_cur_proto,
                            'proto_id':p,
                            'sample_info':[]})
        for k in range(K):
            idx_repeat = top_k_indices[k,p]
            idx_mod = idx_repeat % N
            # info with idx_repeat
            modality = ID_TO_MODALITY[all_modality_ids_stack[idx_repeat].int().item()] # str
            proto_simi = copy.deepcopy(mm_proto_simi_stack[idx_repeat].detach().cpu().numpy()) # P
            # info with idx_mod
            sample_ids = {k:all_sample_ids[k][idx_mod] for k in all_sample_ids.keys()}
            act_cls = {act:all_act_cls[act][idx_mod].detach().cpu().int().numpy() for act in all_act_cls.keys()}
            content = [f'mean relative var of cur proto: {cur_p_rel_var}', 
                       f'relative var of cur sample: {top_k_relative_var[k,p].item()}', 
                       f'sample ids: {sample_ids}\n', 
                       f'modality: {modality}\n',
                       f'labels: {act_cls}\n', 
                       f'proto_simi: {proto_simi[p]}\n',
                       f'last weights of cur proto: {last_weights_cur_proto}\n',
                       ]
            save_path = os.path.join(save_root, str(p), str(k))
            makedir(save_path)
            write_info_txt(content, 
                           os.path.join(save_path, 'sample_info.txt'))
            explain_info[p]['sample_info'].append({'modality': modality,
                                                    'labels': act_cls,
                                                    'image': None,
                                                    'rel_var': top_k_relative_var[k,p].item(),
                                                    'proto_simi': proto_simi[p],
                                                  })
            # visualize
            if modality == 'img':
                all_img = all_inputs['img']
                img = copy.deepcopy(all_img[idx_mod].detach().cpu().numpy()) # 3 T H W
                if args.model_color_order == 'RGB':
                    img = img[[2,1,0],:]
                img_mean, img_std = img_mean_std_BGR(args.img_norm_mode)  # BGR
                img = recover_norm_imgs(img, img_mean, img_std)  # 3 T H W
                img = img.transpose(1,2,3,0).astype(np.int32)  # T H W 3
                # 2D case
                if 'deeplab' in args.img_backbone_name or 'vit' in args.img_backbone_name:
                    img = img[-1:]  # 1 H W 3
                # get feature map
                feat = copy.deepcopy(all_feat['img'][idx_mod].detach().cpu())  # C (T) H W
                if len(feat.shape) == 3:
                    feat = feat.unsqueeze(1)  # C 1 H W
                feat = feat.numpy()
                feat = feat.numpy().transpose(1,2,3,0)  # 1 H W C
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
                cv2.imwrite(os.path.join(save_path, 'mean_max_t.png'), mean_overlay_imgs[max_t])
                if P <= 100:
                    explain_info[p]['sample_info'][k]['image'] = mean_overlay_imgs[max_t]
                else:
                    explain_info[p]['sample_info'][k]['image'] = os.path.join(save_path, 'mean_max_t.png')
            elif modality == 'ctx':
                all_img = all_inputs['ctx']
                img = copy.deepcopy(all_img[idx_mod].detach().cpu().numpy()) # 3/4 (T) H W
                if 'ped_graph' in args.ctx_format:
                    img = img[:3] # 4 T H W -> 3 T H W
                # RGB --> BGR
                if args.model_color_order == 'RGB':
                    img = img[[2,1,0]]
                img_mean, img_std = img_mean_std_BGR(args.img_norm_mode)  # BGR
                img = recover_norm_imgs(img, img_mean, img_std)  # 3 T H W
                img = img.transpose(1,2,3,0).astype(np.int32)  # T H W 3
                # 2D case
                if 'deeplab' in args.ctx_backbone_name or 'vit' in args.ctx_backbone_name:
                    img = img[-1:]  # 1 H W 3
                # get feature map
                feat = copy.deepcopy(all_feat['ctx'][idx_mod].detach().cpu())  # C (T) H W
                if len(feat.shape) == 3:
                    feat = feat.unsqueeze(1)  # C 1 H W
                feat = feat.numpy().transpose(1,2,3,0)  # 1 H W C
                save_path = os.path.join(save_root, str(p), str(k), 'ctx')
                makedir(save_path)
                mean_dir = os.path.join(save_path, 'mean')
                makedir(mean_dir)
                max_dir = os.path.join(save_path, 'max')
                makedir(max_dir)
                min_dir = os.path.join(save_path, 'min')
                makedir(min_dir)
                ori_dir = os.path.join(save_path, 'ori')
                makedir(ori_dir)
                cv2.imwrite(os.path.join(ori_dir, 'ori.png'), img[-1])
                mean_mean, mean_max, mean_min, mean_overlay_imgs, heatmaps = visualize_featmap3d(feat,img, mode='mean', save_dir=mean_dir)
                max_mean, max_max, max_min, _, _ = visualize_featmap3d(feat,img, mode='max', save_dir=max_dir)
                min_mean, min_max, min_min, _, _ = visualize_featmap3d(feat,img, mode='min', save_dir=min_dir)
                # write_info_txt([mean_mean, mean_max, mean_min, max_mean, max_max, max_min, min_mean, min_max, min_min],
                #                os.path.join(save_path, 'feat_info.txt'))
                max_t = np.argmax(np.max(heatmaps, axis=(1,2,3)))
                cv2.imwrite(os.path.join(save_path, 'mean_max_t.png'), mean_overlay_imgs[max_t])
                if P <= 100:
                    explain_info[p]['sample_info'][k]['image'] = mean_overlay_imgs[max_t]
                else:
                    explain_info[p]['sample_info'][k]['image'] = os.path.join(save_path, 'mean_max_t.png')
            elif modality == 'sklt':
                traj = copy.deepcopy(all_inputs['traj_ori'][idx_mod].detach().cpu().int().numpy())  # T 4(int)
                sklt_coords = copy.deepcopy(all_inputs['sklt'][idx_mod].detach().cpu().numpy())  # 2 T nj
                feat = copy.deepcopy(all_feat['sklt'][idx_mod].detach().cpu().numpy())  # obslen*nj
                save_path = os.path.join(save_root, str(p), str(k), 'sklt')
                makedir(save_path)
                if 'coord' in args.sklt_format and 'transformer' in args.sklt_backbone_name:
                    nd, obslen, nj = sklt_coords.shape
                    set_id = all_sample_ids['set_id_int'][idx_mod]
                    vid_id = all_sample_ids['vid_id_int'][idx_mod]
                    img_nms = all_sample_ids['img_nm_int'][idx_mod]
                    obj_id = all_sample_ids['ped_id_int'][idx_mod]
                    dataset_name = all_sample_ids['dataset_name'][idx_mod].detach().item()
                    dataset_name = ID_TO_DATASET[dataset_name]  # int --> str
                    # de-normalize
                    if '0-1' in args.sklt_format:
                        sklt_coords = recover_norm_sklt(sklt_coords, dataset_name)  # 2 T nj (int)
                    sklt_coords = sklt_coords.transpose(1,2,0)  # T nj 2
                    feat = feat.reshape(obslen, nj)  # obslen nj
                    max_t = np.argmax(np.max(feat, axis=(1,)))
                    img_path = get_sklt_img_path(dataset_name,
                                                    set_id=set_id,
                                                    vid_id=vid_id,
                                                    obj_id=obj_id,
                                                    img_nm=img_nms[max_t],
                                                    with_sklt=True,
                                                    )
                    sklt_img = cv2.imread(img_path)[None,:,:,:]  # 1 h w 3
                    overlay_imgs, heatmaps = visualize_sklt_with_pseudo_heatmap(sklt_img, 
                                                                                sklt_coords[max_t:max_t+1],  # 1 nj 2,
                                                                                feat[max_t:max_t+1], 
                                                                                traj[max_t:max_t+1],  # 1 4, 
                                                                                dataset_name, 
                                                                                save_path)
                    cv2.imwrite(os.path.join(save_path, 'max_t.png'), overlay_imgs[0])
                    if P <= 100:
                        explain_info[p]['sample_info'][k]['image'] = overlay_imgs[0]
                    else:
                        explain_info[p]['sample_info'][k]['image'] = os.path.join(save_path, 'max_t.png')

            elif modality == 'traj':
                traj = copy.deepcopy(all_inputs['traj_ori'][idx_mod].detach().cpu().int().numpy())  # obslen 4(int)
                feat = copy.deepcopy(all_feat['traj'][idx_mod].detach().cpu().numpy())  # obslen
                save_path = os.path.join(save_root, str(p), str(k), 'traj')
                makedir(save_path)
                if 'transformer' in args.traj_backbone_name:
                    set_id = all_sample_ids['set_id_int'][idx_mod]
                    vid_id = all_sample_ids['vid_id_int'][idx_mod]
                    img_nms = all_sample_ids['img_nm_int'][idx_mod]
                    obj_id = all_sample_ids['ped_id_int'][idx_mod]
                    dataset_name = all_sample_ids['dataset_name'][idx_mod].detach().item()
                    dataset_name = ID_TO_DATASET[dataset_name]  # int --> str
                    bg_img_path = get_ori_img_path(nm_to_dataset[dataset_name],
                                                set_id=set_id,
                                                vid_id=vid_id,
                                                img_nm=img_nms[-1],
                                                )
                    bg_img = cv2.imread(bg_img_path)
                    bg_blank = np.ones_like(bg_img)*220
                    traj_img = draw_boxes_on_img(bg_img, traj)
                    blank_traj_img = draw_boxes_on_img(bg_blank, traj)
                    cv2.imwrite(filename=os.path.join(save_path, 'traj.png'), 
                                img=traj_img)
                    cv2.imwrite(filename=os.path.join(save_path, 'traj_blank_bg.png'), 
                                img=blank_traj_img)
                    if P <= 100:
                        explain_info[p]['sample_info'][k]['image'] = blank_traj_img
                    else:
                        explain_info[p]['sample_info'][k]['image'] = os.path.join(save_path, 'traj_blank_bg.png')
                        
            elif modality == 'ego':
                save_path = os.path.join(save_root, str(p), str(k), 'ego')
                makedir(save_path)
                ego = all_inputs['ego'][idx_mod].detach().cpu().numpy()
                max_ego = all_inputs['ego'].max().item()
                min_ego = all_inputs['ego'].min().item()
                lim = (min_ego, max_ego)
                feat = copy.deepcopy(all_feat['ego'][idx_mod].detach().cpu().numpy())
                ego_img = vis_1d_seq(ego, lim, save_path, weights=None)
                cv2.imwrite(os.path.join(save_path, 'ego.png'), ego_img)
                if P <= 100:
                    explain_info[p]['sample_info'][k]['image'] = ego_img
                else:
                    explain_info[p]['sample_info'][k]['image'] = os.path.join(save_path, 'ego.png')
            elif modality == 'social':
                save_path = os.path.join(save_root, str(p), str(k), 'social')
                makedir(save_path)
                neighbor_bbox = all_inputs['obs_neighbor_bbox'][idx_mod].detach().cpu().int().numpy() # n_neighbor T 4(int)
                traj = copy.deepcopy(all_inputs['traj_ori'][idx_mod].detach().cpu().int().numpy())  # obslen 4(int)
                weights = copy.deepcopy(all_feat['social'][idx_mod].detach().cpu().numpy())  # obslen*n_neighbor
                n_neighbor, obslen, _ = neighbor_bbox.shape
                weights = weights.reshape(obslen, n_neighbor)  # obslen n_neighbor
                set_id = all_sample_ids['set_id_int'][idx_mod]
                vid_id = all_sample_ids['vid_id_int'][idx_mod]
                img_nms = all_sample_ids['img_nm_int'][idx_mod]
                obj_id = all_sample_ids['ped_id_int'][idx_mod]
                dataset_name = all_sample_ids['dataset_name'][idx_mod].detach().item()
                dataset_name = ID_TO_DATASET[dataset_name]  # int --> str
                bg_img_path = get_ori_img_path(nm_to_dataset[dataset_name],
                                            set_id=set_id,
                                            vid_id=vid_id,
                                            img_nm=img_nms[-1],
                                            )
                bg_img = cv2.imread(bg_img_path)
                social_img = visualize_neighbor_bbox(bg_img,traj[-1],neighbor_bbox[:,-1],weights=weights[-1])
                cv2.imwrite(filename=os.path.join(save_path, 'social.png'), 
                                img=social_img)
                if P <= 100:
                    explain_info[p]['sample_info'][k]['image'] = social_img
                else:
                    explain_info[p]['sample_info'][k]['image'] = os.path.join(save_path, 'social.png')

    log(f'Plotting all explanation in {save_root}')
    plot_all_explanation_no_fusion(explain_info, os.path.join(save_root, 'all_explanation.png'))


def plot_all_explanation_no_fusion(explain_info, 
                                   path, 
                                   part_height=300, 
                                   part_width=350, 
                                   img_height=250,
                                   spacing=50):
    '''
    explain_info: P*[{'mean_rel_var':float, 
                      'last_weights':{act:array}, 
                      'proto_id':int,
                      'sample_info':[{'rel_var':float,
                                      'labels':{act:int}, 
                                      'image':array, 
                                      'modality': str, 
                                      'proto_simi':float,
                                      }]
                      }]
    path: str, the path to save the plot
    row_spacing: float, the spacing between rows
    col_spacing: float, the spacing between columns
    '''
    P = len(explain_info)
    K = len(explain_info[0]['sample_info'])

    # rank by mean_rel_var
    explain_info = sorted(explain_info, key=lambda x: x['mean_rel_var'], reverse=True)

    # Calculate the total height and width of the canvas
    total_height = P * (part_height + spacing)
    total_width = (K+2) * (part_width + spacing)
    txt_height = part_height - img_height

    # Create a blank canvas
    canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

    for i, block in enumerate(explain_info):
        block_y_offset = i * (part_height + spacing)
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
            # img
            img_x_offset = spacing + (j+1)*(part_width + spacing)
            img_y_offset = block_y_offset
            if isinstance(sample['image'], str):
                img = cv2.imread(sample['image'])
            else:
                img = sample['image']
            img_resized = resize_image(img, (img_height, part_width), mode='pad', padding_color=(255, 255, 255))
            canvas[img_y_offset:img_y_offset + img_height, img_x_offset:img_x_offset + part_width] = img_resized
            # txt
            txt_x_offset = img_x_offset
            txt_y_offset = img_y_offset + img_height + 10
            try:
                crossing_label = LABEL_TO_CROSSING[int(sample["labels"]['cross'])] if sample["labels"]['cross'] >= 0 else 'None'
                atomic_label = LABEL_TO_ATOMIC_CHOSEN[int(sample["labels"]['atomic'])] if sample["labels"]['atomic'] >= 0 else 'None'

                label_txt = [crossing_label, atomic_label]
            except:
                import pdb;pdb.set_trace()
            sample_txt = [f'  {sample["modality"]}',
                          f'  rel_var: {round(float(sample["rel_var"]),3)}',
                          f'  proto_simi: {round(float(sample["proto_simi"]),3)}',
                          f'  labels: {label_txt}',
                          ]
            for txt in sample_txt:
                cv2.putText(canvas, txt, (txt_x_offset, txt_y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                txt_y_offset += 15
        # Draw last_weights
        weights_y_offset = block_y_offset
        weights_x_offset = (K+1)*(part_width+spacing)
        for act, cls in block['last_weights'].items():
            weights_text = f"{act}:"
            cv2.putText(canvas, weights_text, (weights_x_offset, weights_y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            weights_cls_y_offset = weights_y_offset + 20
            for c in range(len(cls)):
                weights_text = f"  {c}: {round(float(cls[c]),3)}"
                cv2.putText(canvas, weights_text, (weights_x_offset, weights_cls_y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                weights_cls_y_offset += 20
            weights_y_offset += spacing
            weights_x_offset += part_width // len(block['last_weights'])
    
    # Save the final image
    cv2.imwrite(path, canvas)

    return canvas


if __name__ == '__main__':
    pass