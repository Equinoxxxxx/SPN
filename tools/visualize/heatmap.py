import numpy as np
import torch
import torch.nn.functional as F
import cv2
import os
from tools.utils import makedir

def mask_heatmap(imgs: np.ndarray,
                 heatmaps: np.ndarray,
                 color=(0,0,0),
                 max_dim='all',
                 ):
    '''
    imgs: ndarray (...,H,W,3)
    heatmaps: ndarray (...,H,W)
    '''
    if color == 'random':
        canvas = np.random.rand(*imgs.shape)
    else:
        canvas = np.ones(imgs.shape) * np.array(color)
    if max_dim == 'all':
        max_heat = np.max(heatmaps)
    else:
        max_heat = np.max(heatmaps,axis=(-2, -1), keepdims=True)
    # normalize
    heatmaps = heatmaps / max_heat
    imgs = canvas + imgs*np.expand_dims(heatmaps,axis=-1)

    return imgs


def visualize_featmap3d(featmap, ori_input, mode='mean', channel_weights=None, save_dir='', print_flag=False, log=print):
    '''
    when mode != fix_proto:
        featmap: ndarray T1 H1 W1 C1
        ori_input: ndarray T2 H2 W2 C2
        channel_weights: ndarray C1
    when mode == fix_proto:
        featmap: ndarray T1 H1 W1
        ori_input: ndarray T2 H2 W2 C2
    '''
    assert len(ori_input.shape) == 4, (ori_input.shape)
    if mode in ('mean', 'min', 'max', 'fix_proto'):
        if mode == 'mean':
            featmap = np.mean(featmap, axis=3, keepdims=True)  # T1 H1 W1 1
        elif mode == 'min':
            featmap = np.amin(featmap, axis=3, keepdims=True)  # T1 H1 W1 1
        elif mode == 'max':
            featmap = np.amax(featmap, axis=3, keepdims=True)  # T1 H1 W1 1
        elif mode == 'fix_proto':
            featmap = np.expand_dims(featmap, axis=3)  # T1 H1 W1 1
        featmap = torch.from_numpy(featmap).permute(3, 0, 1, 2).contiguous()  # 1 T1 H1 W1
        featmap = torch.unsqueeze(featmap, 0)  # 1 1 T1 H1 W1
        mask = F.interpolate(featmap, size=(ori_input.shape[0], # T
                                            ori_input.shape[1], # H
                                            ori_input.shape[2]  # W
                                            ), mode='trilinear', align_corners=True)  # 1 1 T2 H2 W2
        mask = torch.squeeze(mask, 0).permute(1, 2, 3, 0).numpy()  # T2 H2 W2 1
        assert mask.shape[:3] == ori_input.shape[:3], (mask.shape, ori_input.shape)
        feat_max = np.amax(mask)
        feat_min = np.amin(mask)
        feat_mean = np.mean(mask)
        mask = mask - np.amin(mask)
        mask = mask / (np.amax(mask) + 1e-8)
        overlay_imgs = []
        for i in range(ori_input.shape[0]):
            img = ori_input[i]
            heatmap = cv2.applyColorMap(np.uint8(255*mask[i]), cv2.COLORMAP_JET)
            heatmap = 0.3*heatmap + 0.5*img
            cv2.imwrite(os.path.join(save_dir,
                                     'feat_heatmap' + str(i) + '.png'),
                        heatmap)
            overlay_imgs.append(heatmap)
        overlay_imgs = np.stack(overlay_imgs, axis=0) # T H W 3
        return feat_mean, feat_max, feat_min, overlay_imgs, mask
    elif mode == 'separate':
        T1, H1, W1, C1 = featmap.shape
        featmap = torch.from_numpy(featmap).permute(3, 0, 1, 2).contiguous()  # C T1 H1 W1
        featmap = torch.unsqueeze(featmap, 0)  # 1 C T1 H1 W1
        mask = torch.nn.functional.interpolate(featmap, size=(ori_input.shape[0], # T
                                                              ori_input.shape[1], # H
                                                              ori_input.shape[2]  # W
                                                              ), mode='trilinear')  # 1 C T2 H2 W2
        mask = mask.permute(2, 3, 4, 1, 0).numpy()  # T2 H2 W2 C 1
        feat_max = np.amax(mask)
        feat_min = np.amin(mask)
        feat_mean = np.mean(mask)
        mask = mask - np.amin(mask)
        mask = mask / (np.amax(mask) + 1e-8)
        for i in range(ori_input.shape[0]):
            img = ori_input[i]
            save_dir_t = os.path.join(save_dir, str(i))
            makedir(save_dir_t)
            for c in range(mask.shape[3]):
                heatmap = cv2.applyColorMap(np.uint8(255*mask[i, :, :, c]), cv2.COLORMAP_JET)
                heatmap = 0.3*heatmap + 0.5*img
                cv2.imwrite(os.path.join(save_dir_t,
                                        'feat_heatmap_channel' + str(c) + '.png'),
                            heatmap)
        return feat_mean, feat_max, feat_min
    elif mode == 'weighted':
        T1, H1, W1, C1 = featmap.shape
        channel_weights = np.expand_dims(channel_weights, axis=(0, 1, 2))  # 1 1 1 C1
        featmap_ = np.mean(featmap * channel_weights, axis=3, keepdims=True)  # T1 H1 W1 1
        featmap_ = torch.from_numpy(featmap_).permute(3, 0, 1, 2).contiguous()  # 1 T1 H1 W1
        featmap_ = torch.unsqueeze(featmap_, 0)  # 1 1 T1 H1 W1
        mask = F.interpolate(featmap_, size=(ori_input.shape[0], # T
                                                              ori_input.shape[1], # H
                                                              ori_input.shape[2]  # W
                                                              ), mode='trilinear', align_corners=True)  # 1 1 T2 H2 W2
        mask = torch.squeeze(mask, 0).permute(1, 2, 3, 0).numpy()  # T2 H2 W2 1
        # mask = scipy.ndimage.zoom(featmap, zoom=[ori_input.shape[0] / featmap.shape[0], 
        #                                          ori_input.shape[1] / featmap.shape[1],
        #                                          ori_input.shape[2] / featmap.shape[2],
        #                                          1]
        #                           )  # T2 H2 W2 1
        assert mask.shape[:3] == ori_input.shape[:3], (mask.shape, ori_input.shape)
        feat_max = np.amax(mask)
        feat_min = np.amin(mask)
        feat_mean = np.mean(mask)
        mask = mask - np.amin(mask)
        mask = mask / (np.amax(mask) + 1e-8)
        for i in range(ori_input.shape[0]):
            img = ori_input[i]
            heatmap = cv2.applyColorMap(np.uint8(255*mask[i]), cv2.COLORMAP_JET)
            overlay = 0.3*heatmap + 0.5*img
            cv2.imwrite(os.path.join(save_dir,
                                     'feat_heatmap' + str(i) + '.png'),
                        overlay)
        channel_weights_path = os.path.join(save_dir, 'channel_weights.txt')
        max_idx = np.argmax(channel_weights[0, 0, 0])
        content = [str(max_idx), str(channel_weights[0, 0, 0, max_idx]), str(channel_weights)]
        with open(channel_weights_path, 'w') as f:
            f.writelines(str(content))
        mask_info_path = os.path.join(save_dir, mode + '_mask_info.txt')
        content = ['max', str(feat_max), ' min', str(feat_min), ' mean', feat_mean, ' ori shape', str([T1, H1, W1, C1])]
        with open(mask_info_path, 'w') as f:
            f.writelines(str(content))
        if print_flag and False:
            # log('channel weight' + str(channel_weights))
            max_idx = np.argmax(channel_weights[0, 0, 0])
            log('max channel idx' + str(max_idx))
            log('max channel weight' + str(channel_weights[0, 0, 0, max_idx]))
            log('weights sum' + str(np.sum(channel_weights)))
            log('ori featmap t0' + str(featmap_.shape))
            log(str(featmap_[0, 0, 0]))
            print(feat_mean, feat_max, feat_min)
        return feat_mean, feat_max, feat_min
    else:
        raise NotImplementedError(mode)

if __name__ == '__main__':
    pass