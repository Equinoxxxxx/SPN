import os
from config import dataset_root


def get_img_path(dataset,
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
        img_root = os.path.join(dataset_root, '/PIE_dataset/images')
        set_id = 'set'+str(set_id).zfill(2)
        vid_id = 'video_'+str(vid_id).zfill(4)
        img_nm = str(img_nm).zfill(5)+'.png'
        img_path = os.path.join(img_root, set_id, vid_id, 'images', img_nm)
    elif dataset.dataset_name == 'JAAD':
        img_root = os.path.join(dataset_root, '/JAAD/images')
        vid_id = 'video_'+str(vid_id).zfill(4)
        img_nm = str(img_nm).zfill(5)+'.png'
        img_path = os.path.join(img_root, vid_id, 'images', img_nm)
    elif dataset.dataset_name == 'nuscenes':
        nusc_sensor = dataset.sensor
        img_root = os.path.join(dataset_root, '/nusc/samples', nusc_sensor)
        sam_id = str(vid_id)
        samtk = dataset.sample_id_to_token[sam_id]
        sam = dataset.nusc.get('sample', samtk)
        sen_data = dataset.nusc.get('sample_data', sam['data'][nusc_sensor])
        img_nm = sen_data['filename']
        img_path = os.path.join(img_root, img_nm)
    elif dataset.dataset_name == 'bdd100k':
        img_root = os.path.join(dataset_root, '/BDD100k/bdd100k/images/track')
        vid_nm = dataset.vid_id2nm[vid_id]
        img_nm = vid_nm + '-' + str(img_nm).zfill(7) + '.jpg'
        for subset in ('train', 'val'):
            img_path = os.path.join(img_root, subset, vid_nm, img_nm)
            if os.path.exists(img_path):
                break
    return img_path
