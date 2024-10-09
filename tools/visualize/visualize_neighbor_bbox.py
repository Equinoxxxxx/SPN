import numpy as np
import cv2

def apply_colormap(weights, colormap=cv2.COLORMAP_JET):
    # Normalize weights to range [0, 255]
    norm_weights = cv2.normalize(weights, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # seqlen,1
    norm_weights = norm_weights[:,0]  # seqlen,
    # Apply colormap
    colored_weights = cv2.applyColorMap(norm_weights, colormap) # seq_len, 1, 3
    colored_weights = colored_weights.transpose(1,0,2) # 1, seqlen, 3
    return colored_weights


def visualize_neighbor_bbox(img, bbox, neighbor_bboxes, weights=None):
    '''
    img: ndarray H W 3
    box: ndarray 4 (ltrb)
    neighbor_bboxes: K 4 (ltrb)
    weights: K
    '''
    # Make a copy of the image to draw on
    img_copy = img.copy()
    
    # Draw the main bbox
    l, t, r, b = bbox
    cv2.rectangle(img_copy, (l, t), (r, b), color=(0, 0, 255), thickness=2)

    if weights is not None:
        # normalize weights
        weights  = weights - np.amin(weights)
        weights = weights / np.amax(weights)
        # Apply colormap to weights
        colored_weights = apply_colormap(weights) # 1, K, 3
    # Draw the neighbor bboxes
    for i, (l, t, r, b) in enumerate(neighbor_bboxes):
        # skip if the bbox is empty
        if l+r+b+t == 0:
            continue
        cv2.rectangle(img_copy, (l, t), (r, b), color=(255, 0, 0), thickness=2)
        if weights is not None:
            color = tuple(int(c) for c in colored_weights[0, i, :])
            # print(colored_weights.shape, color, bbox)
            overlay = img_copy.copy()
            cv2.rectangle(overlay, (l, t), (r, b), color=color, thickness=-1)
            # Add transparency to the filled rectangle
            cv2.addWeighted(overlay, 0.4, img_copy, 0.6, 0, img_copy)
        
    return img_copy
