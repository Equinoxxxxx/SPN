import torch
import numpy as np
from tqdm import tqdm
import os

from tools.utils import makedir, write_info_txt
from tools.datasets.identify_sample import get_img_path
from tools.datasets.dataset_id import ID2DATASET
from tools.data.normalize import recover_norm_imgs, img_mean_std_BGR
from tools.visualize.heatmap import visualize_featmap3d

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
    all_inputs, all_targets, all_info, all_outputs, all_batch_size = forwad_pass(dataloader,
                                                                                model_parallel,
                                                                                device,
                                                                                modalities)
    batch_size = all_batch_size[0]
    n_samples = sum(all_batch_size)
    all_proto_simi = torch.cat([out['proto_simi'] for out in all_outputs], dim=0)  # n_samples P
    simi_mean = all_proto_simi.mean(dim=0)  # (P)
    simi_var = all_proto_simi.var(dim=0, unbiased=True)  # (P)
    all_relative_var = (all_proto_simi - simi_mean.unsqueeze(0))**2 / (simi_var.unsqueeze(0) + 1e-5)  # (n_samples, P)
    if args.topk_metric_explain == 'activation':
        data_to_rank = all_proto_simi
    elif args.topk_metric_explain == 'relative_var':
        data_to_rank = all_relative_var
    top_k_values, top_k_indices = torch.topk(data_to_rank, args.topk_explain, dim=0)  # (k, P) (k, P)
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
    # act cls
    all_act_cls = {k:[] for k in all_targets[0].keys()}
    for target in all_targets:
        for k in target.keys():
            all_act_cls[k].append(target[k])
    for k in all_act_cls.keys():
        all_act_cls[k] = torch.cat(all_act_cls[k], dim=0)  # n_samples,
    # save
    tbar = tqdm(range(P), miniters=1)
    for p in tbar:
        for k in range(K):
            sample_idx = top_k_indices[k,p]
            sample_ids = {k:all_sample_ids[k][sample_idx] for k in all_sample_ids.keys()}
            modality_effs = None
            if all_modality_effs is not None:
                modality_effs = {k:all_modality_effs[k][sample_idx] for k in all_modality_effs.keys()}
            act_cls = {k:all_act_cls[k][sample_idx] for k in all_act_cls.keys()}
            content = [f'{sample_ids}\n', f'{modality_effs}\n', f'{act_cls}\n']
            save_path = os.path.join(save_root, str(p), str(k))
            makedir(save_path)
            write_info_txt(content, 
                           os.path.join(save_path, 'sample_info.txt'))

    # img
    if 'img' in modalities:
        all_img = torch.cat([inp['img'] for inp in all_inputs], dim=0)  # n_samples 3 T H W
        selected_imgs = torch.gather(all_img, 0, top_k_indices.view(-1,1,1,1,1).\
                                expand(K*P, 3, all_img.shape[2], all_img.shape[3], all_img.shape[4]))  # K*P 3 T H W
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
        for p in range(P):
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
                mean_mean, mean_max, mean_min = visualize_featmap3d(feat,img, mode='mean', save_dir=mean_dir)
                max_mean, max_max, max_min = visualize_featmap3d(feat,img, mode='max', save_dir=max_dir)
                min_mean, min_max, min_min = visualize_featmap3d(feat,img, mode='min', save_dir=min_dir)
                # write_info_txt([mean_mean, mean_max, mean_min, max_mean, max_max, max_min, min_mean, min_max, min_min],
                #                os.path.join(save_path, 'feat_info.txt'))
    # ctx
    if 'ctx' in modalities:
        all_img = torch.cat([inp['ctx'] for inp in all_inputs], dim=0)  # n_samples 3 T H W
        selected_imgs = torch.gather(all_img, 0, top_k_indices.view(-1,1,1,1,1).\
                                expand(K*P, 3, all_img.shape[2], all_img.shape[3], all_img.shape[4]))  # K*P 3 T H W
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
        selected_feat = torch.gather(all_feat, 0, top_k_indices.view(-1,1,1,1,1).\
                                expand(K*P, all_feat.shape[1], all_feat.shape[2], all_feat.shape[3], all_feat.shape[4]))  # K*P C T/1 H W
        selected_feat = selected_feat.permute(0,2,3,4,1).\
            reshape(K,P,all_feat.shape[2], all_feat.shape[3], all_feat.shape[4],all_feat.shape[1])  # K P T/1 H W C
        # visualize and save
        for p in range(P):
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
                mean_mean, mean_max, mean_min = visualize_featmap3d(feat,img, mode='mean', save_dir=mean_dir)
                max_mean, max_max, max_min = visualize_featmap3d(feat,img, mode='max', save_dir=max_dir)
                min_mean, min_max, min_min = visualize_featmap3d(feat,img, mode='min', save_dir=min_dir)
                # write_info_txt([mean_mean, mean_max, mean_min, max_mean, max_max, max_min, min_mean, min_max, min_min],
                #                os.path.join(save_path, 'feat_info.txt'))
                
    # sklt TBD
    if 'sklt' in modalities:
        pass
    # traj TBD
    # ego TBD
    # social TBD