import numpy as np
import cv2


def draw_box(img, box):
    '''
    img: ndarray H W 3
    box: ndarray 4 (ltrb)
    '''
    img = cv2.rectangle(img=img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0, 0, 255), thickness=2)
    return img

def draw_boxes_on_img(img, traj_seq, color='b', interval=1, ids=None):
    '''
    img: ndarray H W 3
    traj_seq: ndarray T 4 (ltrb)
    ids: None or ndarray T
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    seq_len = traj_seq.shape[0]
    for i in range(seq_len-1, -1, -interval):
        r = i / seq_len
        # print('traj type:', type(traj_seq))
        if color == 'r':
            _color = (0, 0, int(255*r))
        else:
            _color = (int(255*r), 0, 0)
        img = cv2.rectangle(img=img, pt1=(int(traj_seq[i, 0]), int(traj_seq[i, 1])), pt2=(int(traj_seq[i, 2]), int(traj_seq[i, 3])), color=_color, thickness=2)
        if ids is not None:
            xy = (int(traj_seq[i, 0]), int(traj_seq[i, 1]))
            img = cv2.putText(img, text=str(int(ids[i])), org=xy, fontFace=font, fontScale=0.5, color=(0,0,255),thickness=2)
    return img



