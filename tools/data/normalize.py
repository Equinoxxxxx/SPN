

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

