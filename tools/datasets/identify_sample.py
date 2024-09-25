import os
from config import dataset_root


def get_ori_img_path(dataset,
                 set_id=None,  # int
                 vid_id=None,  # int
                 img_nm=None,  # int
                 ):
    
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
    set_id = set_id.detach().cpu().int().item()
    vid_id = vid_id.detach().cpu().int().item()
    obj_id = obj_id.detach().cpu().int().item()
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
                                '_'.join(str(set_id),str(vid_id),str(obj_id)),
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
    