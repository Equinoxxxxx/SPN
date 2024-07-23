import cv2
import numpy as np
import os
import pickle
import json
from tqdm import tqdm
from ..utils import makedir
from tools.datasets.nuscenes_dataset import NUSC_ROOT
from nuscenes.nuscenes import NuScenes
from .coord_transform import nusc_3dbbox_to_2dbbox
from ..datasets.pie_data import PIE
from ..datasets.jaad_data import JAAD
from config import dataset_root


def crop_img(img, bbox, resize_mode, target_size=(224, 224)):
    l, t, r, b = list(map(int, bbox))
    cropped = img[t:b, l:r]
    if resize_mode == 'ori':
        resized = cropped
    elif resize_mode == 'resized':
        resized = cv2.resize(cropped, target_size)
    elif resize_mode == 'even_padded':
        h = b-t
        w = r-l
        if h < 0 or w < 0:
            raise ValueError('Box size < 0', h, w)
        if h == 0 or w == 0:
            return None
        if  h > 0 and w > 0 and float(w) / h > float(target_size[0]) / target_size[1]:
            ratio = float(target_size[0]) / w
        else:
            ratio = float(target_size[1]) / h
        new_size = (int(w*ratio), int(h*ratio))
        # print(cropped.shape, l, t, r, b, new_size)

        cropped = cv2.resize(cropped, new_size)
        w_pad = target_size[0] - new_size[0]
        h_pad = target_size[1] - new_size[1]
        l_pad = w_pad // 2
        r_pad = w_pad - l_pad
        t_pad = h_pad // 2
        b_pad = h_pad - t_pad
        resized = cv2.copyMakeBorder(cropped,t_pad,b_pad,l_pad,r_pad,cv2.BORDER_CONSTANT,value=(0, 0, 0))  # t, b, l, r
        assert (resized.shape[1], resized.shape[0]) == target_size
    
    return resized

def crop_ctx(img, 
             bbox, 
             mode, 
             target_size=(224, 224),
             padding_value=127):
    ori_H, ori_W = img.shape[:2]
    l, t, r, b = list(map(int, bbox))
    # crop local context
    x = (l+r) // 2
    y = (t+b) // 2
    h = b-t
    w = r-l
    if h == 0 or w == 0:
        return None
    crop_h = h*2
    crop_w = h*2
    crop_l = max(x-h, 0)
    crop_r = min(x+h, ori_W)
    crop_t = max(y-h, 0)
    crop_b = min(y+h, ori_W)
    if mode == 'local':
        # mask target pedestrian
        rect = np.array([[l, t], [r, t], [r, b], [l, b]])
        masked = cv2.fillConvexPoly(img, rect, (127, 127, 127))
        cropped = masked[crop_t:crop_b, crop_l:crop_r]
    elif mode == 'ori_local':
        cropped = img[crop_t:crop_b, crop_l:crop_r]
    l_pad = max(h-x, 0)
    r_pad = max(x+h-ori_W, 0)
    t_pad = max(h-y, 0)
    b_pad = max(y+h-ori_H, 0)
    cropped = cv2.copyMakeBorder(cropped, 
                                 t_pad, b_pad, l_pad, r_pad, 
                                 cv2.BORDER_CONSTANT, 
                                 value=(padding_value, padding_value, padding_value))
    assert cropped.shape[0] == crop_h and cropped.shape[1] == crop_w, (cropped.shape, (crop_h, crop_w))
    # print(cropped.shape, cropped.dtype, img.shape)
    resized = cv2.resize(np.array(cropped, dtype='uint8'), target_size)

    return resized

def crop_img_PIE_JAAD(resize_mode='even_padded', 
                      target_size=(224, 224), 
                      dataset_name='PIE', 
                      ):
    import os
    if dataset_name == 'PIE':
        pie_jaad_root = os.path.join(dataset_root, 'PIE_dataset')
        data_base = PIE(data_path=pie_jaad_root)
        data_opts = {'normalize_bbox': False,
                         'fstride': 1,
                         'sample_type': 'all',
                         'height_rng': [0, float('inf')],
                         'squarify_ratio': 0,
                         'data_split_type': 'default',  # kfold, random, default. default: set03 for test
                         'seq_type': 'intention',  # crossing , intention
                         'min_track_size': 0,  # discard tracks that are shorter
                         'max_size_observe': 1,  # number of observation frames
                         'max_size_predict': 1,  # number of prediction frames
                         'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
                         'balance': False,  # balance the training and testing samples
                         'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
                         'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
                         'seq_type': 'trajectory',
                         'encoder_input_type': ['bbox', 'obd_speed'],
                         'decoder_input_type': [],
                         'output_type': ['intention_binary', 'bbox']
                         }
    else:
        pie_jaad_root = os.path.join(dataset_root, 'JAAD')
        data_opts = {'fstride': 1,
             'sample_type': 'all',  
	         'subset': 'high_visibility',
             'data_split_type': 'default',
             'seq_type': 'trajectory',
	         'height_rng': [0, float('inf')],
	         'squarify_ratio': 0,
             'min_track_size': 0,
             'random_params': {'ratios': None,
                               'val_data': True,
                               'regen_data': True},
             'kfold_params': {'num_folds': 5, 'fold': 1}}
        data_base = JAAD(data_path=pie_jaad_root)
    cropped_root = os.path.join(pie_jaad_root, 'cropped_images')
    data_dir = os.path.join(cropped_root, resize_mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h')
    makedir(data_dir)

    tracks = data_base.generate_data_trajectory_sequence(image_set='all', 
                                                         **data_opts)  # all: 1842  train 882 
    # 'image', 'ped_id', 'bbox', 'center', 'occlusion', 'obd_speed', 'gps_speed', 'heading_angle', 'gps_coord', 'yrp', 'intention_prob', 'intention_binary'
    num_tracks = len(tracks['image'])
    # ids = []
    # for track in tracks['ped_id']:
    #     ids += track
    # id_set = np.unique(ids)
    # print(len(id_set))

    for i in range(num_tracks):
        cur_pid = tracks['ped_id'][i][0][0]  # [[id], [id], ...]
        ped_dir = os.path.join(data_dir, cur_pid)
        makedir(ped_dir)
        track_len = len(tracks['ped_id'][i])
        for j in range(track_len):
            img_path = tracks['image'][i][j]
            target_path = os.path.join(ped_dir, img_path.split('/')[-1])
            img = cv2.imread(img_path)
            l, t, r, b = tracks['bbox'][i][j]  # l t r b
            l, t, r, b = map(int, [l, t, r, b])
            cropped = img[t:b, l:r]
            if resize_mode == 'ori':
                resized = cropped
            elif resize_mode == 'resized':
                resized = cv2.resize(cropped, target_size)
            elif resize_mode == 'even_padded':
                h = b-t
                w = r-l
                if  float(w) / h > float(target_size[0]) / target_size[1]:
                    ratio = float(target_size[0]) / w
                else:
                    ratio = float(target_size[1]) / h
                new_size = (int(w*ratio), int(h*ratio))
                cropped = cv2.resize(cropped, new_size)
                w_pad = target_size[0] - new_size[0]
                h_pad = target_size[1] - new_size[1]
                l_pad = w_pad // 2
                r_pad = w_pad - l_pad
                t_pad = h_pad // 2
                b_pad = h_pad - t_pad
                resized = cv2.copyMakeBorder(cropped,
                                             t_pad,b_pad,l_pad,r_pad,
                                             cv2.BORDER_CONSTANT,
                                             value=(0, 0, 0))  # t, b, l, r
                assert (resized.shape[1], resized.shape[0]) == target_size
            else:
                h = b-t
                w = r-l
                if  float(w) / h > float(target_size[0]) / target_size[1]:
                    ratio = float(target_size[0]) / w
                else:
                    ratio = float(target_size[1]) / h
                new_size = (int(w*ratio), int(h*ratio))
                cropped = cv2.resize(cropped, new_size)
                w_pad = target_size[0] - new_size[0]
                h_pad = target_size[1] - new_size[1]
                resized = cv2.copyMakeBorder(cropped,0,h_pad,0,w_pad,cv2.BORDER_CONSTANT,value=(0, 0, 0))  # t, b, l, r
                assert (resized.shape[1], resized.shape[0]) == target_size
            cv2.imwrite(target_path, resized)
        print(i, ped_dir, 'done')

def crop_ctx_PIE_JAAD(mode='ori_local', target_size=(224, 224), dataset_name='PIE'):
    if dataset_name == 'PIE':
        pie_jaad_root = os.path.join(dataset_root, 'PIE_dataset')
        data_base = PIE(data_path=pie_jaad_root)
        data_opts = {'normalize_bbox': False,
                         'fstride': 1,
                         'sample_type': 'all',
                         'height_rng': [0, float('inf')],
                         'squarify_ratio': 0,
                         'data_split_type': 'default',  # kfold, random, default. default: set03 for test
                         'seq_type': 'intention',  # crossing , intention
                         'min_track_size': 0,  # discard tracks that are shorter
                         'max_size_observe': 1,  # number of observation frames
                         'max_size_predict': 1,  # number of prediction frames
                         'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
                         'balance': False,  # balance the training and testing samples
                         'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
                         'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
                         'seq_type': 'trajectory',
                         'encoder_input_type': ['bbox', 'obd_speed'],
                         'decoder_input_type': [],
                         'output_type': ['intention_binary', 'bbox']
                         }
    else:
        pie_jaad_root = os.path.join(dataset_root, 'JAAD')
        data_opts = {'fstride': 1,
             'sample_type': 'all',  
	         'subset': 'high_visibility',
             'data_split_type': 'default',
             'seq_type': 'trajectory',
	         'height_rng': [0, float('inf')],
	         'squarify_ratio': 0,
             'min_track_size': 0,
             'random_params': {'ratios': None,
                               'val_data': True,
                               'regen_data': True},
             'kfold_params': {'num_folds': 5, 'fold': 1}}
        data_base = JAAD(data_path=pie_jaad_root)
    context_root = os.path.join(pie_jaad_root, 'context')
    makedir(context_root)
    data_dir = os.path.join(context_root, mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h')
    makedir(data_dir)

    tracks = data_base.generate_data_trajectory_sequence(image_set='all', **data_opts)
    num_tracks = len(tracks['image'])
    mask_value = (127, 127, 127)

    if mode == 'mask_ped':
        for i in range(num_tracks):
            cur_pid = tracks['ped_id'][i][0][0]  # [[id], [id], ...]
            ped_dir = os.path.join(data_dir, cur_pid)
            makedir(ped_dir)
            track_len = len(tracks['ped_id'][i])
            for j in range(track_len):
                img_path = tracks['image'][i][j]
                target_path = os.path.join(ped_dir, img_path.split('/')[-1])
                img = cv2.imread(img_path)
                l, t, r, b = tracks['bbox'][i][j]  # l t r b
                l, t, r, b = map(int, [l, t, r, b])
                # if mode == 'mask_ped':
                rect = np.array([[l, t], [r, t], [r, b], [l, b]])
                masked = cv2.fillConvexPoly(img, rect, mask_value)
                resized = cv2.resize(masked, target_size)
                cv2.imwrite(target_path, resized)
            print(i, ped_dir, 'done')

    elif mode == 'ori':
        for i in range(num_tracks):
            cur_pid = tracks['ped_id'][i][0][0]  # [[id], [id], ...]
            ped_dir = os.path.join(data_dir, cur_pid)
            makedir(ped_dir)
            track_len = len(tracks['ped_id'][i])
            for j in range(track_len):
                img_path = tracks['image'][i][j]
                target_path = os.path.join(ped_dir, img_path.split('/')[-1])
                img = cv2.imread(img_path)
                resized = cv2.resize(img, target_size)
                cv2.imwrite(target_path, resized)
            print(i, ped_dir, 'done')

    elif mode == 'ori_local':
        for i in range(num_tracks):
            cur_pid = tracks['ped_id'][i][0][0]  # [[id], [id], ...]
            ped_dir = os.path.join(data_dir, cur_pid)
            makedir(ped_dir)
            track_len = len(tracks['ped_id'][i])
            for j in range(track_len):
                img_path = tracks['image'][i][j]
                target_path = os.path.join(ped_dir, img_path.split('/')[-1])
                img = cv2.imread(img_path)
                ori_H, ori_W = img.shape[0], img.shape[1]
                l, t, r, b = tracks['bbox'][i][j]  # l t r b
                l, t, r, b = map(int, [l, t, r, b])
                # crop local context
                x = (l+r) // 2
                y = (t+b) // 2
                h = b-t
                w = r-l
                crop_h = h*2
                crop_w = h*2
                crop_l = max(x-h, 0)
                crop_r = min(x+h, ori_W)
                crop_t = max(y-h, 0)
                crop_b = min(y+h, ori_W)
                cropped = img[crop_t:crop_b, crop_l:crop_r]
                l_pad = max(h-x, 0)
                r_pad = max(x+h-ori_W, 0)
                t_pad = max(h-y, 0)
                b_pad = max(y+h-ori_H, 0)
                cropped = cv2.copyMakeBorder(cropped, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, value=mask_value)
                assert cropped.shape[0] == crop_h and cropped.shape[1] == crop_w, (cropped.shape, (crop_h, crop_w))
                resized = cv2.resize(cropped, target_size)
                cv2.imwrite(target_path, resized)
            print(i, ped_dir, 'done')
    elif mode == 'local':
        for i in range(num_tracks):
            cur_pid = tracks['ped_id'][i][0][0]  # [[id], [id], ...]
            ped_dir = os.path.join(data_dir, cur_pid)
            makedir(ped_dir)
            track_len = len(tracks['ped_id'][i])
            for j in range(track_len):
                img_path = tracks['image'][i][j]
                target_path = os.path.join(ped_dir, img_path.split('/')[-1])
                img = cv2.imread(img_path)
                ori_H, ori_W = img.shape[0], img.shape[1]
                l, t, r, b = tracks['bbox'][i][j]  # l t r b
                l, t, r, b = map(int, [l, t, r, b])
                # mask target pedestrian
                rect = np.array([[l, t], [r, t], [r, b], [l, b]])
                masked = cv2.fillConvexPoly(img, rect, mask_value)
                # crop local context
                x = (l+r) // 2
                y = (t+b) // 2
                h = b-t
                w = r-l
                crop_h = h*2
                crop_w = h*2
                crop_l = max(x-h, 0)
                crop_r = min(x+h, ori_W)
                crop_t = max(y-h, 0)
                crop_b = min(y+h, ori_W)
                cropped = masked[crop_t:crop_b, crop_l:crop_r]
                l_pad = max(h-x, 0)
                r_pad = max(x+h-ori_W, 0)
                t_pad = max(h-y, 0)
                b_pad = max(y+h-ori_H, 0)
                cropped = cv2.copyMakeBorder(cropped, 
                                             t_pad, b_pad, l_pad, r_pad, 
                                             cv2.BORDER_CONSTANT, 
                                             value=mask_value)
                assert cropped.shape[0] == crop_h and \
                    cropped.shape[1] == crop_w, \
                    (cropped.shape, (crop_h, crop_w))
                resized = cv2.resize(cropped, target_size)
                cv2.imwrite(target_path, resized)
            print(i, ped_dir, 'done')

def crop_img_TITAN(tracks, resize_mode='even_padded', target_size=(224, 224), obj_type='p'):
    crop_root = os.path.join(dataset_root, 'TITAN/TITAN_extra/cropped_images')
    makedir(crop_root)
    data_root = os.path.join(dataset_root, 'TITAN/honda_titan_dataset/dataset')
    if obj_type == 'p':
        crop_obj_path = os.path.join(crop_root, resize_mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h', 'ped')
        makedir(crop_obj_path)
    else:
        crop_obj_path = os.path.join(crop_root, resize_mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h', 'veh')
        makedir(crop_obj_path)
    for i in tqdm(range(len(tracks['clip_id']))):
        cid = int(tracks['clip_id'][i][0])
        oid = int(float(tracks['obj_id'][i][0]))
        cur_clip_path = os.path.join(crop_obj_path, str(cid))
        makedir(cur_clip_path)
        cur_obj_path = os.path.join(cur_clip_path, str(oid))
        makedir(cur_obj_path)
        
        for j in range(len(tracks['clip_id'][i])):
            img_nm = tracks['img_nm'][i][j]
            l, t, r, b = list(map(int, tracks['bbox'][i][j]))
            img_path = os.path.join(data_root, 'images_anonymized', 'clip_'+str(cid), 'images', img_nm)
            tgt_path = os.path.join(cur_obj_path, img_nm)
            img = cv2.imread(img_path)
            cropped = img[t:b, l:r]
            if resize_mode == 'ori':
                resized = cropped
            elif resize_mode == 'resized':
                resized = cv2.resize(cropped, target_size)
            elif resize_mode == 'even_padded':
                h = b-t
                w = r-l
                if  float(w) / h > float(target_size[0]) / target_size[1]:
                    ratio = float(target_size[0]) / w
                else:
                    ratio = float(target_size[1]) / h
                new_size = (int(w*ratio), int(h*ratio))
                cropped = cv2.resize(cropped, new_size)
                w_pad = target_size[0] - new_size[0]
                h_pad = target_size[1] - new_size[1]
                l_pad = w_pad // 2
                r_pad = w_pad - l_pad
                t_pad = h_pad // 2
                b_pad = h_pad - t_pad
                resized = cv2.copyMakeBorder(cropped,t_pad,b_pad,l_pad,r_pad,cv2.BORDER_CONSTANT,value=(0, 0, 0))  # t, b, l, r
                assert (resized.shape[1], resized.shape[0]) == target_size
            else:
                raise NotImplementedError(resize_mode)
            cv2.imwrite(tgt_path, resized)
        print(i, cid, cur_obj_path, 'done')

def crop_ctx_TITAN(tracks, mode='ori_local', target_size=(224, 224), obj_type='p'):
    ori_H, ori_W = 1520, 2704
    crop_root = os.path.join(dataset_root, 'TITAN/TITAN_extra/context')
    makedir(crop_root)
    data_root = os.path.join(dataset_root, 'TITAN/honda_titan_dataset/dataset')
    if obj_type == 'p':
        crop_obj_path = os.path.join(crop_root, mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h', 'ped')
        makedir(crop_obj_path)
    else:
        crop_obj_path = os.path.join(crop_root, mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h', 'veh')
        makedir(crop_obj_path)
    
    if mode == 'local':
        for i in range(len(tracks['clip_id'])):  # tracks
            cid = int(tracks['clip_id'][i][0])
            oid = int(float(tracks['obj_id'][i][0]))
            cur_clip_path = os.path.join(crop_obj_path, str(cid))
            makedir(cur_clip_path)
            cur_obj_path = os.path.join(cur_clip_path, str(oid))
            makedir(cur_obj_path)
            for j in range(len(tracks['clip_id'][i])):  # time steps in each track
                img_nm = tracks['img_nm'][i][j]
                l, t, r, b = list(map(int, tracks['bbox'][i][j]))
                img_path = os.path.join(data_root, 
                                        'images_anonymized', 
                                        'clip_'+str(cid), 
                                        'images', 
                                        img_nm)
                tgt_path = os.path.join(cur_obj_path, 
                                        img_nm)
                img = cv2.imread(img_path)
                # mask target pedestrian
                rect = np.array([[l, t], [r, t], [r, b], [l, b]])
                masked = cv2.fillConvexPoly(img, rect, (127, 127, 127))
                # crop local context
                x = (l+r) // 2
                y = (t+b) // 2
                h = b-t
                w = r-l
                crop_h = h*2
                crop_w = h*2
                crop_l = max(x-h, 0)
                crop_r = min(x+h, ori_W)
                crop_t = max(y-h, 0)
                crop_b = min(y+h, ori_W)
                cropped = masked[crop_t:crop_b, crop_l:crop_r]
                l_pad = max(h-x, 0)
                r_pad = max(x+h-ori_W, 0)
                t_pad = max(h-y, 0)
                b_pad = max(y+h-ori_H, 0)
                cropped = cv2.copyMakeBorder(cropped, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, value=(127, 127, 127))
                assert cropped.shape[0] == crop_h and cropped.shape[1] == crop_w, (cropped.shape, (crop_h, crop_w))
                resized = cv2.resize(cropped, target_size)
                cv2.imwrite(tgt_path, resized)
            print(i, cid, oid, cur_obj_path, 'done')
    elif mode == 'ori_local':
        for i in range(len(tracks['clip_id'])):  # tracks
            cid = int(tracks['clip_id'][i][0])
            oid = int(float(tracks['obj_id'][i][0]))
            cur_clip_path = os.path.join(crop_obj_path, str(cid))
            makedir(cur_clip_path)
            cur_obj_path = os.path.join(cur_clip_path, str(oid))
            makedir(cur_obj_path)
            for j in range(len(tracks['clip_id'][i])):  # time steps in each track
                img_nm = tracks['img_nm'][i][j]
                l, t, r, b = list(map(int, tracks['bbox'][i][j]))
                img_path = os.path.join(data_root, 'images_anonymized', 'clip_'+str(cid), 'images', img_nm)
                tgt_path = os.path.join(cur_obj_path, img_nm)
                img = cv2.imread(img_path)
                # crop local context
                x = (l+r) // 2
                y = (t+b) // 2
                h = b-t
                w = r-l
                crop_h = h*2
                crop_w = h*2
                crop_l = max(x-h, 0)
                crop_r = min(x+h, ori_W)
                crop_t = max(y-h, 0)
                crop_b = min(y+h, ori_W)
                cropped = img[crop_t:crop_b, crop_l:crop_r]
                l_pad = max(h-x, 0)
                r_pad = max(x+h-ori_W, 0)
                t_pad = max(h-y, 0)
                b_pad = max(y+h-ori_H, 0)
                cropped = cv2.copyMakeBorder(cropped, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, value=(127, 127, 127))
                assert cropped.shape[0] == crop_h and cropped.shape[1] == crop_w, (cropped.shape, (crop_h, crop_w))
                resized = cv2.resize(cropped, target_size)
                cv2.imwrite(tgt_path, resized)
            print(i, cid, oid, cur_obj_path, 'done')
    else:
        raise NotImplementedError(mode)

def crop_img_ctx_nusc(obj_type='ped', sensor='CAM_FRONT', modality='img', resize_mode='even_padded', target_size=(224, 224)):
    print(f'{obj_type}, {sensor}, {modality}, {resize_mode}')
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)

    f = open(os.path.join(dataset_root, 'nusc/extra/trainval_token_to_sample_id.pkl'), 'rb')
    token_to_sample_id = pickle.load(f)
    f.close()

    f = open(os.path.join(dataset_root, 'nusc/extra/trainval_token_to_instance_id.pkl', 'rb'))
    token_to_instance_id = pickle.load(f)
    f.close()

    f = open(os.path.join(dataset_root, 'nusc/extra', '_'.join(['anns_train', obj_type, sensor])+'.pkl'), 'rb')
    instk_to_anntks = pickle.load(f)
    f.close()
    f = open(os.path.join(dataset_root, 'nusc/extra', '_'.join(['anns_val', obj_type, sensor])+'.pkl'), 'rb')
    val_instk_to_anntks = pickle.load(f)
    f.close()
    instk_to_anntks.update(val_instk_to_anntks)

    if modality == 'img':
        root = os.path.join(dataset_root, 'nusc/extra/cropped_images')
        root = os.path.join(root, sensor, resize_mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h', obj_type)
        makedir(root)
    elif modality == 'ctx':
        root = os.path.join(dataset_root, 'nusc/extra/context')
        root = os.path.join(root, sensor, resize_mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h', obj_type)
        makedir(root)

    for instk in tqdm(instk_to_anntks):
        insid = token_to_instance_id[instk]
        ins_path = os.path.join(root, insid)
        makedir(ins_path)
        anntk_seqs = instk_to_anntks[instk]
        seq_all = []
        for seq in anntk_seqs:
            seq_all += seq
        for anntk in seq_all:
            bbox = nusc_3dbbox_to_2dbbox(nusc, anntk)[0]  # ltrb
            l, t, r, b = bbox
            if r-l<2 or b-t<2:
                continue
            ann = nusc.get('sample_annotation', anntk)
            samtk = nusc.get('sample', ann['sample_token'])['token']
            sam = nusc.get('sample', samtk)
            samid = token_to_sample_id[samtk]
            sen_data = nusc.get('sample_data', sam['data'][sensor])
            img_path = os.path.join(NUSC_ROOT, sen_data['filename'])
            img = cv2.imread(img_path)
            if modality == 'img':
                cropped_img = crop_img(img, bbox, resize_mode, target_size)
            elif modality == 'ctx':
                cropped_img = crop_ctx(img, bbox, resize_mode, target_size)
            if cropped_img is None:
                continue
            save_path = os.path.join(ins_path, samid+'.png')
            cv2.imwrite(save_path, cropped_img)

def crop_img_ctx_bdd100k(data_root=os.path.join(dataset_root, 'BDD100k/bdd100k'),
                     sub_set='train_val',
                     resize_mode='even_padded',
                     ctx_format='ori_local',
                     img_size=(224, 224)):
    vid_nm2id_path = os.path.join(data_root, 'extra', 'vid_nm2id.pkl')
    with open(vid_nm2id_path, 'rb') as f:
        vid_nm2id = pickle.load(f)
    img_root = os.path.join(data_root, 'images', 'track')
    label_root = os.path.join(data_root, 'labels', 'box_track_20')
    crop_root = os.path.join(data_root, 
                             'extra', 
                             'cropped_images', 
                             resize_mode,
                             f'{img_size[0]}w_by_{img_size[1]}h')
    makedir(crop_root)
    if ctx_format != '':
        ctx_root = os.path.join(data_root, 
                                'extra', 
                                'context', 
                                ctx_format,
                                f'{img_size[0]}w_by_{img_size[1]}h')
        makedir(ctx_root)
    for _subset in reversed(sub_set.split('_')):
        print(f'Processing {_subset} set')
        label_dir = os.path.join(label_root, _subset)
        # traverse video
        for lfnm in tqdm(os.listdir(label_dir)):
            l_path = os.path.join(label_dir, lfnm)
            with open(l_path) as f:
                vid_l = json.load(f)  # list
            vid_nm = lfnm.replace('.json', '')
            vid_id = vid_nm2id[vid_nm]
            # traverse label
            for img_l in vid_l:
                img_nm = img_l['name']
                img_id = img_nm.split('-')[-1].replace('.jpg', '')
                img_id_int = int(img_id)
                for l in img_l['labels']:
                    cls = l['category']
                    oid = l['id']
                    oid_int = int(oid)
                    cls_k = 'ped' if cls in ('other person', 'pedestrian', 'rider') else 'veh'
                    ltrb = [l['box2d']['x1'],
                            l['box2d']['y1'],
                            l['box2d']['x2'],
                            l['box2d']['y2']]
                    img = cv2.imread(os.path.join(img_root, 
                                                  _subset,
                                                  vid_nm,
                                                  img_nm))
                    # save cropped images
                    tgt_crop_dir = os.path.join(crop_root,
                                           cls_k,
                                           str(oid_int),
                                           )
                    makedir(tgt_crop_dir)
                    tgt_crop_path = os.path.join(tgt_crop_dir,
                                                 str(img_id_int)+'.png')
                    cropped = crop_img(img,
                                       ltrb,
                                       resize_mode,
                                       img_size)
                    if cropped is not None:
                        cv2.imwrite(tgt_crop_path, cropped)
                    # save ctx
                    if ctx_format != '':
                        tgt_ctx_dir = os.path.join(ctx_root,
                                                    cls_k,
                                                    str(oid_int),
                                                    )
                        makedir(tgt_ctx_dir)
                        tgt_ctx_path = os.path.join(tgt_ctx_dir,
                                                    str(img_id_int)+'.png')
                        cropped = crop_ctx(img,
                                        ltrb,
                                        ctx_format,
                                        img_size)
                        if cropped is not None:
                            cv2.imwrite(tgt_ctx_path, cropped)

if __name__ == '__main__':
    crop_img_ctx_bdd100k(img_size=(224, 224))
    crop_img_ctx_bdd100k(ctx_format='', img_size=(288, 384))