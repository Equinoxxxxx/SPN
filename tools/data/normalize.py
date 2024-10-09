DATASET_TO_IMG_SIZE = {
        'TITAN': (1520, 2704),
        'PIE': (1080, 1920),
        'JAAD': (1080, 1920),
        'nuscenes': (900, 1600),
        'bdd100k': (720, 1280),
    }

def img_mean_std_BGR(norm_mode):
    # BGR order
    if norm_mode == 'activitynet':
        # mean = [0.4477, 0.4209, 0.3906]
        # std = [0.2767, 0.2695, 0.2714]
        mean = [0.3906, 0.4209, 0.4477]
        std = [0.2714, 0.2695, 0.2767]
    elif norm_mode == 'kinetics':
        # mean = [0.4345, 0.4051, 0.3775]
        # std = [0.2768, 0.2713, 0.2737]
        mean = [0.3775, 0.4051, 0.4345]
        std = [0.2737, 0.2713, 0.2768]
    elif norm_mode == '0.5' or norm_mode == 'tf':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif norm_mode == 'torch':
        mean = [0.406, 0.456, 0.485]  # BGR
        std = [0.225, 0.224, 0.229]
    
    elif norm_mode == 'ori':
        mean = None
        std = None
    
    return mean, std

def norm_imgs(imgs, means, stds):
    '''
    imgs: torch.tensor: C (T) H W
    means: list: [B mean, G mean, R mean]
    stds: list: [B std, G std, R std]
    '''
    # if len(imgs.size()) == 4:
    #     C, T, H, W = imgs.size()
    # elif len(imgs.size()) == 3:
    #     C, H, W = imgs.size()
    # else:
    #     raise ValueError(imgs.size())
    
    imgs = imgs / 255.
    imgs[0] = imgs[0] - means[0]
    imgs[1] = imgs[1] - means[1]
    imgs[2] = imgs[2] - means[2]
    imgs[0] = imgs[0] / stds[0]
    imgs[1] = imgs[1] / stds[1]
    imgs[2] = imgs[2] / stds[2]

    return imgs

def recover_norm_imgs(imgs, means, stds):
    '''
    imgs: torch.tensor: 3 ...
    means: list: [B mean, G mean, R mean]
    '''
    imgs[0] = imgs[0] * stds[0]
    imgs[1] = imgs[1] * stds[1]
    imgs[2] = imgs[2] * stds[2]
    imgs[0] = imgs[0] + means[0]
    imgs[1] = imgs[1] + means[1]
    imgs[2] = imgs[2] + means[2]
    imgs = imgs * 255.

    return imgs

def norm_sklt(sklt, dataset_name):
    '''
    sklt: 2 (T nj)
    '''
    sklt[0] = sklt[0] / DATASET_TO_IMG_SIZE[dataset_name][1]
    sklt[1] = sklt[1] / DATASET_TO_IMG_SIZE[dataset_name][0]

    return sklt

def recover_norm_sklt(sklt, dataset_name):
    '''
    sklt: tensor 2 (T nj) (xy)
    '''
    sklt[0] = sklt[0] * DATASET_TO_IMG_SIZE[dataset_name][1]
    sklt[1] = sklt[1] * DATASET_TO_IMG_SIZE[dataset_name][0]

    return sklt

def sklt_local_to_global(sklt, bbox, sklt_img_size=(384,288)):
    '''
    sklt: 2 T nj (xy)
    bbox: T 4 (ltrb)
    '''
    ys = (bbox[:, 1] + bbox[:, 3]) / 2 # T
    xs = (bbox[:, 0] + bbox[:, 2]) / 2 # T
    hs = bbox[:, 3] - bbox[:, 1] # T
    ws = bbox[:, 2] - bbox[:, 0] # T
    
    sklt[1] = (sklt[1]/sklt_img_size[0]-0.5) * hs[:, None] + ys[:, None]
    sklt[0] = (sklt[0]/sklt_img_size[1]-0.5) * ws[:, None] + xs[:, None]

    return sklt

def sklt_global_to_local_warning(sklt, bbox, sklt_img_size=(384,288)):
    '''
    sklt: 2 T nj
    bbox: T 4 (ltrb)
    '''
    xs = (bbox[:, 0] + bbox[:, 2]) / 2 # T
    ys = (bbox[:, 1] + bbox[:, 3]) / 2 # T
    ws = bbox[:, 2] - bbox[:, 0] # T
    hs = bbox[:, 3] - bbox[:, 1] # T

    T = sklt.shape[1]
    for t in range(T):
        if hs[t] <= 0:
            hs[t] = 1
            print(f'non-positive hs at {t}: {hs[t]}')
            raise ValueError()
        if ws[t] <= 0:
            ws[t] = 1
            print(f'non-positive ws at {t}: {ws[t]}')
            raise ValueError()
    sklt[1] = (sklt[1] - ys[:, None]) / hs[:, None] * sklt_img_size[0] + sklt_img_size[0]/2
    sklt[0] = (sklt[0] - xs[:, None]) / ws[:, None] * sklt_img_size[1] + sklt_img_size[1]/2

    return sklt

def sklt_global_to_local(sklt, bbox, sklt_img_size=(384,288)):
    '''
    sklt: 2 T nj
    bbox: T 4 (ltrb)
    '''
    xs = (bbox[:, 0] + bbox[:, 2]) / 2 # T
    ys = (bbox[:, 1] + bbox[:, 3]) / 2 # T
    ws = bbox[:, 2] - bbox[:, 0] # T
    hs = bbox[:, 3] - bbox[:, 1] # T

    T = sklt.shape[1]
    for t in range(T):
        if hs[t] <= 0:
            print(f'non-positive hs at {t}: {hs[t]}')
            hs[t] = 1
            
        if ws[t] <= 0:
            print(f'non-positive ws at {t}: {ws[t]}')
            ws[t] = 1
            
    sklt[1] = (sklt[1] - ys[:, None]) / hs[:, None] * sklt_img_size[0] + sklt_img_size[0]/2
    sklt[0] = (sklt[0] - xs[:, None]) / ws[:, None] * sklt_img_size[1] + sklt_img_size[1]/2

    return sklt

def norm_bbox(bbox, dataset_name):
    '''
    bbox: T 4 (ltrb)
    '''
    bbox[:, 0] = bbox[:, 0] / DATASET_TO_IMG_SIZE[dataset_name][1]
    bbox[:, 1] = bbox[:, 1] / DATASET_TO_IMG_SIZE[dataset_name][0]
    bbox[:, 2] = bbox[:, 2] / DATASET_TO_IMG_SIZE[dataset_name][1]
    bbox[:, 3] = bbox[:, 3] / DATASET_TO_IMG_SIZE[dataset_name][0]

    return bbox

def recover_norm_bbox(bbox, dataset_name):
    '''
    bbox: torch.tensor T 4
    '''
    bbox[:, 0] = bbox[:, 0] * DATASET_TO_IMG_SIZE[dataset_name][1]
    bbox[:, 1] = bbox[:, 1] * DATASET_TO_IMG_SIZE[dataset_name][0]
    bbox[:, 2] = bbox[:, 2] * DATASET_TO_IMG_SIZE[dataset_name][1]
    bbox[:, 3] = bbox[:, 3] * DATASET_TO_IMG_SIZE[dataset_name][0]

    return bbox