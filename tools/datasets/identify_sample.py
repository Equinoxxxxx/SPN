import os
from config import dataset_root
import torch

ID_TO_DATASET={
    0: 'PIE',
    1: 'JAAD',
    2: 'TITAN',
    3: 'nuscenes',
    4: 'bdd100k',
}

DATASET_TO_ID={
    'PIE': 0,
    'JAAD': 1,
    'TITAN': 2,
    'nuscenes': 3,
    'bdd100k': 4,
}

MODALITY_TO_ID = {
    'img': 0,
    'sklt': 1,
    'ctx': 2,
    'traj': 3,
    'ego': 4,
    'social': 5,
}

ID_TO_MODALITY = {
    0: 'img',
    1: 'sklt',
    2: 'ctx',
    3: 'traj',
    4: 'ego',
    5: 'social',
}

LABEL_TO_CROSSING = {
    0: 'not crossing',
    1: 'crossing',
}

LABEL_TO_ATOMIC_CHOSEN = {
    0: 'standing',
    1: 'running',
    2: 'bending',
    3: 'walking',
    4: 'sitting',
    5: 'none of above',
}

LABEL_TO_SIMPLE_CONTEXTUAL = {
    0: 'crossing a street at pedestrian crossing',
    1: 'jaywalking',
    2: 'waiting to cross street',
    3: 'motorcycling',
    4: 'biking',
    5: 'walking along the side of the road',
    6: 'walking on the road',
    7: 'cleaning an object',
    8: 'closing',
    9: 'opening',
    10: 'exiting a building',
    11: 'entering a building',
    12: 'none of above',
}

LABEL_TO_COMPLEX_CONTEXTUAL = {
    0: 'unloading',
    1: 'loading',
    2: 'getting in 4 wheel vehicle',
    3: 'getting out of 4 wheel vehicle',
    4: 'getting on 2 wheel vehicle',
    5: 'getting off 2 wheel vehicle',
    6: 'none of above',
}

LABEL_TO_COMMUNICATIVE = {
    0: 'looking into phone',
    1: 'talking on phone',
    2: 'talking in group',
    3: 'none of above',
}

LABEL_TO_TRANSPORTIVE = {
    0: 'pushing',
    1: 'carrying with both hands',
    2: 'pulling',
    3: 'none of above',
}

LABEL_TO_AGE = {
    0: 'child',
    1: 'adult',
    2: 'senior',
}

def get_ori_img_path(dataset,
                 set_id=None,  # torch.tensor
                 vid_id=None,  # torch.tensor
                 img_nm=None,  # torch.tensor
                 ):
    set_id = set_id.detach().cpu().int().item()
    vid_id = vid_id.detach().cpu().int().item()
    img_nm = img_nm.detach().cpu().int().item()
    if dataset.dataset_name == 'TITAN':
        img_root = os.path.join(dataset_root, 'TITAN/honda_titan_dataset/dataset/images_anonymized')
        vid_id = 'clip_'+str(vid_id)
        img_nm = str(img_nm).zfill(6)+'.png'
        img_path = os.path.join(img_root, vid_id, 'images', img_nm)
    elif dataset.dataset_name == 'PIE':
        img_root = os.path.join(dataset_root, 'PIE_dataset/images')
        set_id = 'set'+str(set_id).zfill(2)
        vid_id = 'video_'+str(vid_id).zfill(4)
        img_nm = str(img_nm).zfill(5)+'.png'
        img_path = os.path.join(img_root, set_id, vid_id, img_nm)
    elif dataset.dataset_name == 'JAAD':
        img_root = os.path.join(dataset_root, 'JAAD/images')
        vid_id = 'video_'+str(vid_id).zfill(4)
        img_nm = str(img_nm).zfill(5)+'.png'
        img_path = os.path.join(img_root, vid_id, img_nm)
    elif dataset.dataset_name == 'nuscenes':
        nusc_sensor = dataset.sensor
        img_root = os.path.join(dataset_root, 'nusc')
        sam_id = str(img_nm)
        samtk = dataset.sample_id_to_token[sam_id]
        sam = dataset.nusc.get('sample', samtk)
        sen_data = dataset.nusc.get('sample_data', sam['data'][nusc_sensor])
        img_nm = sen_data['filename']
        img_path = os.path.join(img_root, img_nm)
    elif dataset.dataset_name == 'bdd100k':
        img_root = os.path.join(dataset_root, 'BDD100k/bdd100k/images/track')
        vid_nm = dataset.vid_id2nm[vid_id]
        img_nm = vid_nm + '-' + str(img_nm).zfill(7) + '.jpg'
        for subset in ('train', 'val'):
            img_path = os.path.join(img_root, subset, vid_nm, img_nm)
            if os.path.exists(img_path):
                break
    return img_path

def get_sklt_img_path(dataset_name,
                 set_id=None,  # torch.tensor
                 vid_id=None,  # torch.tensor
                 obj_id=None,  # torch.tensor
                 img_nm=None,  # torch.tensor
                 with_sklt=True,
                 ):
    if isinstance(set_id, torch.Tensor):
        set_id = set_id.detach().cpu().int().item()
    if isinstance(vid_id, torch.Tensor):
        vid_id = vid_id.detach().cpu().int().item()
    if isinstance(obj_id, torch.Tensor):
        obj_id = obj_id.detach().cpu().int().item()
    if isinstance(img_nm, torch.Tensor):
        img_nm = img_nm.detach().cpu().int().item()
    dataset_to_extra_root = {
        'bdd100k': 'BDD100k/bdd100k/extra',
        'nuscenes': 'nusc/extra',
        'TITAN': 'TITAN/TITAN_extra',
        'PIE': 'PIE_dataset',
    }
    interm_dir = 'sk_vis' if with_sklt else 'cropped_images'
    extra_dir = dataset_to_extra_root[dataset_name]
    sklt_img_root = os.path.join(dataset_root,
                                 extra_dir,
                                 interm_dir,
                                 'even_padded/288w_by_384h'
                                 )
    if dataset_name in ('TITAN', 'bdd100k', 'nuscenes') and\
        not with_sklt:
        sklt_img_root = os.path.join(sklt_img_root, 'ped')
    if dataset_name == 'TITAN':
        img_path = os.path.join(sklt_img_root,
                                str(vid_id),
                                str(obj_id),
                                str(img_nm).zfill(6)+'.png')
    elif dataset_name == 'PIE':
        img_path = os.path.join(sklt_img_root,
                                '_'.join([str(set_id),str(vid_id),str(obj_id)]),
                                str(img_nm).zfill(5)+'.png')
    elif dataset_name == 'nuscenes':
        img_path = os.path.join(sklt_img_root,
                                str(obj_id),
                                str(img_nm)+'.png')
    elif dataset_name == 'bdd100k':
        img_path = os.path.join(sklt_img_root,
                                str(obj_id),
                                str(img_nm)+'.png')
    else:
        raise NotImplementedError(dataset_name)

    return img_path                                
    