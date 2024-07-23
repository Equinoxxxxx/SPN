import numpy as np
import cv2

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

if __name__ == '__main__':
    pass