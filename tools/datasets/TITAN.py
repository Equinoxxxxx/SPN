import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import pickle
from re import T
from turtle import resizemode
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

import time
import copy
import os
from tqdm import tqdm
import pdb
import csv

from .pie_data import PIE
from .jaad_data import JAAD
from ..utils import makedir
from ..utils import mapping_20, ltrb2xywh, coord2pseudo_heatmap, TITANclip_txt2list, cls_weights
from ..utils import get_random_idx
from ..data.normalize import img_mean_std_BGR, norm_imgs, sklt_local_to_global, norm_bbox, norm_sklt
from ..data.transforms import RandomHorizontalFlip, RandomResizedCrop, crop_local_ctx
from torchvision.transforms import functional as TVF
from .dataset_id import DATASET_TO_ID, ID_TO_DATASET
from ..data.bbox import ltrb2xywh_seq, ltrb2xywh_multi_seq, bbox2d_relation_multi_seq, pad_neighbor
from config import dataset_root


ATOM_ACTION_LABEL_ORI = {  # 3 no samples; 7, 8 no test samples
    'standing': 0,
    'running': 1,
    'bending': 2,
    'kneeling': 3,
    'walking': 4,
    'sitting': 5,
    'squatting': 6,
    'jumping': 7,
    'laying down': 8,
    'none of the above': 9,
    # '': 9
}

ATOM_ACTION_LABEL_CORRECTED1 = {  # combine kneel, jump, lay down and none of the above
    'standing': 0,
    'running': 1,
    'bending': 2,
    'walking': 3,
    'sitting': 4,
    'squatting': 5,

    'kneeling': 6,
    'jumping': 6,
    'laying down': 6,
    'none of the above': 6,
    # '': 6
}

ATOM_ACTION_LABEL_CORRECTED2 = {  # remove kneel
    'standing': 0,
    'running': 1,
    'bending': 2,

    'walking': 3,
    'sitting': 4,
    'squatting': 5,
    'jumping': 6,
    'laying down': 7,
    'none of the above': 8,
}

ATOM_ACTION_LABEL_CHOSEN = {  # combine kneel, jump, lay down, squat and none of the above
    'standing': 0,
    'running': 1,
    'bending': 2,
    'walking': 3,
    'sitting': 4,

    'squatting': 5,
    'kneeling': 5,
    'jumping': 5,
    'laying down': 5,
    'none of the above': 5,
    # '': 6
}

ATOM_ACTION_LABEL_CHOSEN2 = {  # combine kneel, jump, lay down, squat and none of the above
    'standing': 0,
    'running': 1,
    'bending': 2,
    'walking': 3,
    'sitting': 4,

    # 'squatting': 5,
    # 'kneeling': 5,
    # 'jumping': 5,
    # 'laying down': 5,
    # 'none of the above': 5,
    # '': 6
}

SIMPLE_CONTEXTUAL_LABEL = {
    'crossing a street at pedestrian crossing': 0,
    'jaywalking (illegally crossing NOT at pedestrian crossing)': 1,
    'waiting to cross street': 2,
    'motorcycling': 3,
    'biking': 4,
    'walking along the side of the road': 5,
    'walking on the road': 6,
    'cleaning an object': 7,
    'closing': 8,
    'opening': 9,
    'exiting a building': 10,
    'entering a building': 11,
    'none of the above': 12,
    # '': 12
}

COMPLEX_CONTEXTUAL_LABEL = {
    'unloading': 0,
    'loading': 1,
    'getting in 4 wheel vehicle': 2,
    'getting out of 4 wheel vehicle': 3,
    'getting on 2 wheel vehicle': 4,
    'getting off 2 wheel vehicle': 5,
    'none of the above': 6,
    # '': 6
}

COMMUNICATIVE_LABEL = {
    'looking into phone': 0,
    'talking on phone': 1,
    'talking in group': 2,
    'none of the above': 3,
    # '': 3
}

TRANSPORTIVE_LABEL = {
    'pushing': 0,
    'carrying with both hands': 1,
    'pulling': 2,
    'none of the above': 3,
    # '': 3
}

MOTOIN_STATUS_LABEL = {
    'stopped': 0,
    'moving': 1,
    'parked': 2,
    'none of the above': 3,
    # '': 3
}

AGE_LABEL = {
    'child': 0,
    'adult': 1,
    'senior over 65': 2,
    # '': 3
}

LABEL2COLUMN = {
    'img_nm': 0,
    'obj_type': 1,
    'obj_id': 2,
    'trunk_open': 7,
    'motion_status': 8,
    'doors_open': 9,
    'communicative': 10,
    'complex_context': 11,
    'atomic_actions': 12,
    'simple_context': 13,
    'transporting': 14,
    'age': 15
}



ATOM_ACTION_LABEL = ATOM_ACTION_LABEL_CHOSEN

LABEL2DICT = {
    'atomic_actions': ATOM_ACTION_LABEL,
    'simple_context': SIMPLE_CONTEXTUAL_LABEL,
    'complex_context': COMPLEX_CONTEXTUAL_LABEL,
    'communicative': COMMUNICATIVE_LABEL,
    'transporting': TRANSPORTIVE_LABEL,
    'age': AGE_LABEL,
}

NUM_CLS_ATOMIC = max([ATOM_ACTION_LABEL[k] for k in ATOM_ACTION_LABEL]) + 1
NUM_CLS_SIMPLE = max([SIMPLE_CONTEXTUAL_LABEL[k] for k in SIMPLE_CONTEXTUAL_LABEL]) + 1
NUM_CLS_COMPLEX = max([COMPLEX_CONTEXTUAL_LABEL[k] for k in COMPLEX_CONTEXTUAL_LABEL]) + 1
NUM_CLS_COMMUNICATIVE = max([COMMUNICATIVE_LABEL[k] for k in COMMUNICATIVE_LABEL]) + 1
NUM_CLS_TRANSPORTING = max([TRANSPORTIVE_LABEL[k] for k in TRANSPORTIVE_LABEL]) + 1
NUM_CLS_AGE = max([AGE_LABEL[k] for k in AGE_LABEL]) + 1

OCC_NUM = 0

LABEL_2_KEY = {
    'crossing': 'cross',
    'atomic_actions': 'atomic',
    'complex_context': 'complex',
    'communicative': 'communicative',
    'transporting': 'transporting',
}

KEY_2_LABEL = {
    'cross': 'crossing',
    'atomic': 'atomic_actions',
    'complex': 'complex_context',
    'communicative': 'communicative',
    'transporting': 'transporting',
}

LABEL_2_IMBALANCE_CLS = {
    'crossing': [1],
    'atomic_actions': [1, 2, 4, 5],
    'complex_context': [0, 1, 2, 3, 4, 5],
    'communicative': [0, 1, 2],
    'transporting': [0, 1, 2],
}

ACT_SET_TO_N_CLS = {
    'cross': 2,
    'atomic': NUM_CLS_ATOMIC,
    'simple': NUM_CLS_SIMPLE,
    'complex': NUM_CLS_COMPLEX,
    'communicative': NUM_CLS_COMMUNICATIVE,
    'transporting': NUM_CLS_TRANSPORTING,
    'age': NUM_CLS_AGE,
}


# video id --> img id
#          --> ped id
class TITAN_dataset(Dataset):
    def __init__(self,
                 sub_set='default_train',
                 track_save_path='',
                 offset_traj=False,
                 neighbor_mode='last_frame',
                 obs_len=4, pred_len=4, overlap_ratio=0.5, recog_act=0,
                 obs_fps=2,
                 target_color_order='BGR', img_norm_mode='torch',
                 required_labels=['atomic_actions', 'simple_context'], 
                 multi_label_cross=0, 
                 act_sets=['cross'],
                 loss_weight='sklearn',
                 tte=None,
                 small_set=0,
                 resize_mode='even_padded', crop_size=(224, 224),
                 modalities=['img', 'sklt', 'ctx', 'traj', 'ego', 'social'],
                 img_format='',
                 ctx_format='local', ctx_size=(224, 224),
                 sklt_format='coord',
                 traj_format='ltrb',
                 ego_format='accel',
                 social_format='rel_loc',
                 augment_mode='random_hflip',
                 seg_cls=['person', 'vehicle', 'road', 'traffic_light'],
                 max_n_neighbor=10,
                 pop_occl_track=1,
                 min_wh=(72,36),
                 ) -> None:
        super(Dataset, self).__init__()
        self.img_size = (1520, 2704)
        self.fps = 10
        self.dataset_name = 'TITAN'
        print(f'------------------Init{self.dataset_name}------------------')
        self.sub_set = sub_set
        self.offset_traj = offset_traj
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_interval = self.fps // obs_fps - 1
        self.overlap_ratio = overlap_ratio
        self.recog_act = recog_act
        self.obs_or_pred = 'obs' if self.recog_act else 'pred'
        self.target_color_order = target_color_order
        self.img_norm_mode = img_norm_mode
        self.img_mean, self.img_std = img_mean_std_BGR(self.img_norm_mode)
        # sequence length considering interval
        self._obs_len = self.obs_len * (self.seq_interval + 1)
        self._pred_len = self.pred_len * (self.seq_interval + 1)

        self.modalities = modalities
        self.resize_mode = resize_mode
        self.crop_size = crop_size
        self.img_format = img_format
        self.ctx_format = ctx_format
        self.ctx_size = ctx_size
        self.sklt_format = sklt_format
        self.traj_format = traj_format
        self.ego_format = ego_format
        self.social_format = social_format
        self.track_save_path = track_save_path
        self.required_labels = required_labels
        self.multi_label_cross = multi_label_cross
        self.use_cross = 'cross' in act_sets
        self.use_atomic = 'atomic' in act_sets
        self.use_simple = 'simple' in act_sets
        self.use_complex = 'complex' in act_sets
        self.use_communicative = 'communicative' in act_sets
        self.use_transporting = 'transporting' in act_sets
        self.use_age = 'age' in act_sets
        self.loss_weight = loss_weight
        self.neighbor_mode = neighbor_mode
        self.tte = tte
        self.small_set = small_set
        self.augment_mode = augment_mode
        self.seg_cls = seg_cls
        self.pop_occl_track = pop_occl_track
        self.max_n_neighbor = max_n_neighbor
        self.transforms = {'random': 0,
                            'balance': 0,
                            'hflip': None,
                            'resized_crop': {'img': None,
                                            'ctx': None,
                                            'sklt': None}}

        self.ori_root = os.path.join(dataset_root, 'TITAN/honda_titan_dataset/dataset')
        self.extra_root = os.path.join(dataset_root, 'TITAN/TITAN_extra')
        self.cropped_img_root = os.path.join(self.extra_root,
                                             'cropped_images', 
                                             self.resize_mode, 
                                             str(crop_size[1])+'w_by_'\
                                                +str(crop_size[0])+'h')
        if self.ctx_format in ('ped_graph', 'ped_graph_seg'):
            ctx_format_dir = 'ori_local'
        else:
            ctx_format_dir = self.ctx_format
        self.ctx_root = os.path.join(self.extra_root, 
                                     'context', 
                                     ctx_format_dir, 
                                     str(ctx_size[1])+'w_by_'\
                                        +str(ctx_size[0])+'h')
        self.ped_ori_local_root = os.path.join(dataset_root, 'TITAN/TITAN_extra/context/ori_local/224w_by_224h/ped')
        self.sk_vis_path = os.path.join(dataset_root, 'TITAN/TITAN_extra/sk_vis/even_padded/288w_by_384h/')
        self.sk_coord_path = os.path.join(dataset_root, 'TITAN/TITAN_extra/sk_coords/even_padded/288w_by_384h/')
        self.sk_heatmap_path = os.path.join(dataset_root, 'TITAN/TITAN_extra/sk_heatmaps/even_padded/288w_by_384h/')
        self.sk_p_heatmap_path = os.path.join(dataset_root, 'TITAN/TITAN_extra/sk_pseudo_heatmaps/even_padded/48w_by_48h/')
        self.seg_root = os.path.join(dataset_root, 'TITAN/TITAN_extra/seg_sam')
        if self.sub_set == 'default_train':
            clip_txt_path = os.path.join(dataset_root, 'TITAN/honda_titan_dataset/dataset/train_set.txt')
        elif self.sub_set == 'default_val':
            clip_txt_path = os.path.join(dataset_root, 'TITAN/honda_titan_dataset/dataset/val_set.txt')
        elif self.sub_set == 'default_test':
            clip_txt_path = os.path.join(dataset_root, 'TITAN/honda_titan_dataset/dataset/test_set.txt')
        elif self.sub_set == 'all':
            clip_txt_path = os.path.join(dataset_root, 'TITAN/honda_titan_dataset/titan_clips.txt')
        else:
            raise NotImplementedError(self.sub_set)
        self.imgnm_to_objid_path = os.path.join(self.extra_root, 
                                                'imgnm_to_objid_to_ann.pkl')

        # load clip ids
        self.clip_id_list = TITANclip_txt2list(clip_txt_path)  # list of str
        # load tracks info
        if os.path.exists(self.track_save_path):
            with open(self.track_save_path, 'rb') as f:
                track_info = pickle.load(f)
            self.ids = track_info['ids']
            self.p_tracks = track_info['p_tracks']
            self.num_p_tracks = track_info['num_p_tracks']
            self.v_tracks = track_info['v_tracks']
            self.num_v_tracks = track_info['num_v_tracks']
        else:
            annos, self.ids = self.add_cid()
            self.p_tracks, self.num_p_tracks = self.get_p_tracks(annos)
            # import pdb; pdb.set_trace()
            self.v_tracks, self.num_v_tracks = self.get_v_tracks(annos)
            track_info = {
                'ids': self.ids,
                'p_tracks': self.p_tracks,
                'num_p_tracks': self.num_p_tracks,
                'v_tracks': self.v_tracks,
                'num_v_tracks': self.num_v_tracks
            }

            track_f_nm = 'neighbors.pkl'
            if self.neighbor_mode:
                track_f_nm = 'w_' + track_f_nm
            else:
                track_f_nm = 'wo_' + track_f_nm
            track_save_path = os.path.join(self.extra_root, 
                                           'saved_tracks', 
                                           self.sub_set, 
                                           track_f_nm)
            with open(track_save_path, 'wb') as f:
                pickle.dump(track_info, f)

        # crop_imgs(self.p_tracks, resize_mode=img_mode, target_size=img_size, obj_type='p')
        # print('Crop done')
        # return
        # get cid to img name to obj id dict
        if not os.path.exists(self.imgnm_to_objid_path) or True:
            self.imgnm_to_objid = \
                self.get_imgnm_to_objid(self.p_tracks, 
                                        self.v_tracks, 
                                        self.imgnm_to_objid_path)
        else:
            with open(self.imgnm_to_objid_path, 'rb') as f:
                self.imgnm_to_objid = pickle.load(f)
        
        # filter short tracks
        self.p_tracks_filtered, self.num_p_tracks = \
            self.filter_short_tracks(self.p_tracks, 
                                     self._obs_len+self._pred_len)
        # split tracks into samples
        self.samples = self.track2sample(self.p_tracks_filtered)

        # get neighbors
        if 'social' in self.modalities:
            self.neighbor_seq_path = os.path.join(self.extra_root,
                f'fps_{self.fps}_obs_{self.obs_len}_pred_{self.pred_len}_interval_{self.seq_interval}_overlap_{self.overlap_ratio}.pkl')
            self.samples = self.get_neighbor_relation(self.samples,
                                                self.neighbor_seq_path)
        # num samples
        self.num_samples = len(self.samples['obs']['img_nm'])

        # apply interval
        if self.seq_interval > 0:
            self.downsample_seq()
            
        # small set
        if small_set > 0:
            small_set_size = int(self.num_samples * small_set)
            for k in self.samples['obs'].keys():
                self.samples['obs'][k] = self.samples['obs'][k]\
                    [:small_set_size]
            for k in self.samples['pred'].keys():
                self.samples['pred'][k] = self.samples['pred'][k]\
                    [:small_set_size]
            self.num_samples = small_set_size

        # cross or not
        obs_cross_labels = self.multi2binary(self.samples['obs']\
                                             ['simple_context'], 
                                             [0, 1])
        self.samples['obs']['crossing'] = obs_cross_labels
        pred_cross_labels = self.multi2binary(self.samples['pred']\
                                              ['simple_context'], 
                                              [0, 1])
        self.samples['pred']['crossing'] = pred_cross_labels
        print('num samples: ', self.num_samples)

        # augmentation
        self.samples = self._add_augment(self.samples)

        # class count
        print(self.sub_set, 
              'pred crossing', 
              len(self.samples[self.obs_or_pred]['crossing']), 
              self.num_samples, 
              self.samples[self.obs_or_pred]['crossing'][-1])
        self.n_c = np.sum(np.array(self.samples[self.obs_or_pred]['crossing'])\
                          [:, -1])
        print('self.n_c', self.n_c)
        self.n_nc = self.num_samples - self.n_c
        self.num_samples_cls = [self.n_nc, self.n_c]
        self.class_weights = {}
        if self.multi_label_cross:
            labels = np.array(self.samples[self.obs_or_pred]\
                                ['simple_context'])[:, -1]
            # print(labels.shape, labels)
            self.num_samples_cls = []
            for i in range(13):
                n_cur_cls = sum(labels == i)
                self.num_samples_cls.append(n_cur_cls)
            
            print('label distr', self.num_samples, self.num_samples_cls)
        print('self.num_samples_cls', self.num_samples_cls)
        self.class_weights['cross'] = cls_weights(self.num_samples_cls, 
                                                  'sklearn')

        if self.use_atomic:
            labels = np.array(self.samples['pred']['atomic_actions'])[:, -1]
            self.num_samples_atomic = []
            for i in range(NUM_CLS_ATOMIC):
                n_cur_cls = sum(labels == i)
                self.num_samples_atomic.append(n_cur_cls)
            print('atomic label distr', 
                  self.num_samples, 
                  self.num_samples_atomic)
            self.class_weights['atomic'] = cls_weights(self.num_samples_atomic, 
                                                       'sklearn')
        if self.use_simple:
            labels = np.array(self.samples['pred']['simple_context'])[:, -1]
            self.num_samples_simple = []
            for i in range(NUM_CLS_SIMPLE):
                n_cur_cls = sum(labels == i)
                self.num_samples_simple.append(n_cur_cls)
            print('simple label distr', 
                  self.num_samples, 
                  self.num_samples_simple)
            self.class_weights['simple'] = cls_weights(self.num_samples_simple, 
                                                       'sklearn')
        if self.use_complex:
            labels = np.array(self.samples['pred']['complex_context'])[:, -1]
            self.num_samples_complex = []
            for i in range(NUM_CLS_COMPLEX):
                n_cur_cls = sum(labels == i)
                self.num_samples_complex.append(n_cur_cls)
            assert sum(self.num_samples_complex) == self.num_samples, \
                (sum(self.num_samples_complex), self.num_samples)
            print('complex label distr', 
                  self.num_samples, 
                  self.num_samples_complex)
            self.class_weights['complex'] = \
                cls_weights(self.num_samples_complex, 'sklearn')
        if self.use_communicative:
            labels = np.array(self.samples['pred']['communicative'])[:, -1]
            self.num_samples_communicative = []
            for i in range(NUM_CLS_COMMUNICATIVE):
                n_cur_cls = sum(labels == i)
                self.num_samples_communicative.append(n_cur_cls)
            print('communicative label distr', 
                  self.num_samples, 
                  self.num_samples_communicative)
            self.class_weights['communicative'] = \
                cls_weights(self.num_samples_communicative, 'sklearn')
        if self.use_transporting:
            labels = np.array(self.samples['pred']['transporting'])[:, -1]
            self.num_samples_transporting = []
            for i in range(NUM_CLS_TRANSPORTING):
                n_cur_cls = sum(labels == i)
                self.num_samples_transporting.append(n_cur_cls)
            print('transporting label distr', 
                  self.num_samples, 
                  self.num_samples_transporting)
            self.class_weights['transporting'] = \
                cls_weights(self.num_samples_transporting, 'sklearn')
        if self.use_age:
            labels = np.array(self.samples['pred']['age'])[:, -1]
            self.num_samples_age = []
            for i in range(NUM_CLS_AGE):
                n_cur_cls = sum(labels == i)
                self.num_samples_age.append(n_cur_cls)
            self.class_weights['age'] = \
                cls_weights(self.num_samples_age, 'sklearn')
        
        # apply interval
        if self.seq_interval > 0:
            self.downsample_seq()
            print('Applied interval')
            print('cur input len', len(self.samples['obs']['img_nm_int']))

    def __len__(self):
        return self.num_samples
    
    def rm_small_bb(self, data, min_size):
        print('----------Remove small bb-----------')
        min_w, min_h = min_size
        idx = list(range(self.num_samples))
        new_idx = list(range(self.num_samples))

        bboxes = np.array(data['']['obs_bbox'])  # ltrb
        hws = np.stack([bboxes[:, :, 3] - bboxes[:, :, 1], bboxes[:, :, 2] - bboxes[:, :, 0]], axis=2)  # mean: 134, 46
        print('hws shape: ', hws.shape)
        print('mean h: ', np.mean(hws[:, :, 0]))
        print('mean w: ', np.mean(hws[:, :, 1]))
        for i in idx:
            for hw in hws[i]:
                if hw[0] < min_h or hw[1] < min_w:
                    new_idx.remove(i)
                    break
        
        for k in data.keys():
            data[k] = data[k][new_idx]
        print('n samples before removing small bb', self.num_samples)
        self.num_samples = len(new_idx)
        print('n samples after removing small bb', self.num_samples)

        return data

    def __getitem__(self, idx):
        obs_bbox_offset = copy.deepcopy(torch.tensor(self.samples['obs']['bbox_normed'][idx]).float())  # T 4
        obs_bbox = copy.deepcopy(torch.tensor(self.samples['obs']['bbox'][idx]).float())  # T 4
        pred_bbox_offset = copy.deepcopy(torch.tensor(self.samples['pred']['bbox_normed'][idx]).float())
        pred_bbox = copy.deepcopy(torch.tensor(self.samples['pred']['bbox'][idx]).float())
        obs_ego = torch.tensor(self.samples['obs']['ego_motion'][idx]).float()[:, 0:1]  # [accel, ang_vel] 0 for accel only
        assert len(obs_ego.shape) == 2
        clip_id_int = torch.tensor(int(self.samples['obs']['clip_id'][idx][0]))  # str --> int
        ped_id_int = torch.tensor(int(float(self.samples['obs']['obj_id'][idx][0])))
        img_nm_int = torch.tensor(self.samples['obs']['img_nm_int'][idx])

        obs_bbox[:,2] = torch.clamp(obs_bbox[:,2], min=0, max=self.img_size[1]) # r
        obs_bbox[:,3] = torch.clamp(obs_bbox[:,3], min=0, max=self.img_size[0]) # b
        obs_bbox_ori = copy.deepcopy(obs_bbox)  # T 4
        pred_bbox_ori = copy.deepcopy(pred_bbox)
        # squeeze the coords
        if '0-1' in self.traj_format:
            obs_bbox_offset = norm_bbox(obs_bbox_offset, self.dataset_name)
            pred_bbox_offset = norm_bbox(pred_bbox_offset, self.dataset_name)
            obs_bbox = norm_bbox(obs_bbox, self.dataset_name)
            pred_bbox = norm_bbox(pred_bbox, self.dataset_name)
            for bb in obs_bbox:
                for i in bb:
                    if i > 1:
                        import pdb;pdb.set_trace()
        # act labels
        if self.multi_label_cross:
            target = torch.tensor(self.samples[self.obs_or_pred]\
                                        ['simple_context'][idx][-1])  # int
        else:
            target = torch.tensor(self.samples[self.obs_or_pred]\
                                        ['crossing'][idx][-1])  # int
        simple_context = torch.tensor(self.samples[self.obs_or_pred]\
                                       ['simple_context'][idx][-1])
        atomic_action = torch.tensor(self.samples[self.obs_or_pred]\
                                      ['atomic_actions'][idx][-1])
        complex_context = torch.tensor(self.samples[self.obs_or_pred]\
                                        ['complex_context'][idx][-1])
        communicative = torch.tensor(self.samples[self.obs_or_pred]\
                                      ['communicative'][idx][-1])
        transporting = torch.tensor(self.samples[self.obs_or_pred]\
                                     ['transporting'][idx][-1])
        age = torch.tensor(self.samples[self.obs_or_pred]\
                           ['age'][idx][-1])
        sample = {'dataset_name': torch.tensor(DATASET_TO_ID[self.dataset_name]),
                  'set_id_int': torch.tensor(-1),
                  'vid_id_int': clip_id_int,  # int
                  'ped_id_int': ped_id_int,  # int
                  'img_nm_int': img_nm_int,
                  'obs_bboxes': obs_bbox_offset,
                  'obs_bboxes_unnormed': obs_bbox,
                  'obs_bboxes_ori': obs_bbox_ori,
                  'obs_ego': obs_ego,
                  'pred_act': target,
                  'pred_bboxes': pred_bbox_offset,
                  'pred_bboxes_ori': pred_bbox_ori,
                  'pred_bboxes_unnormed': pred_bbox,
                  'atomic_actions': atomic_action,
                  'simple_context': simple_context,
                  'complex_context': complex_context,  # (1,)
                  'communicative': communicative,
                  'transporting': transporting,
                  'age': age,
                  'hflip_flag': torch.tensor(-1),
                  'img_ijhw': torch.tensor([-1, -1, -1, -1]),
                  'ctx_ijhw': torch.tensor([-1, -1, -1, -1]),
                  'sklt_ijhw': torch.tensor([-1, -1, -1, -1]),
                  'obs_neighbor_relation': torch.zeros((1, self.obs_len, 5)),
                  'obs_neighbor_bbox': torch.zeros((1, self.obs_len, 4)),
                  'obs_neighbor_oid': torch.zeros((1,)),
                  }
        if 'social' in self.modalities:
            # import pdb;pdb.set_trace()
            relations, neighbor_bbox, neighbor_oid =\
                  pad_neighbor([copy.deepcopy(np.array(self.samples['obs']['neighbor_relations'][idx]).transpose(1,0,2)),
                                copy.deepcopy(np.array(self.samples['obs']['neighbor_bbox'][idx]).transpose(1,0,2)),
                                copy.deepcopy(self.samples['obs']['neighbor_oid'][idx])],
                                self.max_n_neighbor)
            if self.social_format == 'rel_loc':
                sample['obs_neighbor_relation'] = torch.tensor(relations).float()  # K T 5
            elif self.social_format == 'ori_traj':
                sample['obs_neighbor_relation'] =\
                      torch.cat([obs_bbox_ori.unsqueeze(0), torch.tensor(neighbor_bbox).float()], 0) # K+1 T 4
            sample['obs_neighbor_bbox'] = torch.tensor(neighbor_bbox).float() # K T 4
            sample['obs_neighbor_oid'] = torch.tensor(neighbor_oid).float() # K
            # print(f'neighbor_bbox: {sample["obs_neighbor_bbox"]}')
        if 'img' in self.modalities:
            imgs = []
            for img_nm in self.samples['obs']['img_nm'][idx]:
                img_path = os.path.join(self.cropped_img_root, 
                                        'ped', 
                                        self.samples['obs']['clip_id']\
                                            [idx][0], 
                                        str(int(float(self.samples['obs']\
                                                      ['obj_id'][idx][0]))), 
                                        img_nm)
                imgs.append(cv2.imread(img_path))
            imgs = np.stack(imgs, axis=0)
            # (T, H, W, C) -> (C, T, H, W)
            ped_imgs = torch.from_numpy(imgs).float().permute(3, 0, 1, 2)
            # normalize img
            if self.img_norm_mode != 'ori':
                ped_imgs = norm_imgs(ped_imgs, self.img_mean, self.img_std)
            # BGR -> RGB
            if self.target_color_order == 'RGB':
                ped_imgs = torch.flip(ped_imgs, dims=[0])
            sample['ped_imgs'] = ped_imgs
        if 'ctx' in self.modalities:
            if self.ctx_format in ('local','ori_local','mask_ped','ori',
                                   'ped_graph', 'ped_graph_seg'):
                ctx_imgs = []
                for img_nm in self.samples['obs']['img_nm'][idx]:
                    img_path = os.path.join(self.ctx_root, 
                                            'ped', 
                                            self.samples['obs']['clip_id']\
                                                [idx][0], 
                                            str(int(float(self.samples\
                                                          ['obs']['obj_id']\
                                                            [idx][0]))), 
                                            img_nm)
                    ctx_imgs.append(cv2.imread(img_path))
                ctx_imgs = np.stack(ctx_imgs, axis=0)
                # (T, H, W, C) -> (C, T, H, W)
                ctx_imgs = torch.from_numpy(ctx_imgs).float().\
                    permute(3, 0, 1, 2)
                # normalize img
                if self.img_norm_mode != 'ori':
                    ctx_imgs = norm_imgs(ctx_imgs, 
                                         self.img_mean, self.img_std)
                # BGR -> RGB
                if self.target_color_order == 'RGB':
                    ctx_imgs = torch.flip(ctx_imgs, dims=[0])
                # add segmentation channel
                if self.ctx_format in ('ped_graph', 'ped_graph_seg'):
                    all_c_seg = []
                    img_nm = self.samples['obs']['img_nm'][idx][-1]
                    vid_dir = self.samples['obs']['clip_id'][idx][0]
                    oid = str(int(float(self.samples['obs']['obj_id'][idx][0])))
                    for c in self.seg_cls:
                        seg_path = os.path.join(self.extra_root,
                                                'cropped_seg',
                                                c,
                                                'ori_local/224w_by_224h',
                                                vid_dir,
                                                'ped',
                                                oid,
                                                img_nm.replace('png', 'pkl'))
                        with open(seg_path, 'rb') as f:
                            segmap = pickle.load(f)*1  # h w int
                        all_c_seg.append(torch.from_numpy(segmap))
                    all_c_seg = torch.stack(all_c_seg, dim=-1)  # h w n_cls
                    all_c_seg = torch.argmax(all_c_seg, dim=-1, keepdim=True).permute(2, 0, 1)  # 1 h w
                    ctx_imgs = torch.concat([ctx_imgs[:, -1], all_c_seg], dim=0).unsqueeze(1)  # 4 1 h w
                sample['obs_context'] = ctx_imgs  # [c, obs_len, H, W]
            elif self.ctx_format in \
                ('seg_ori_local', 'seg_local'):
                # load imgs
                ctx_imgs = []
                for img_nm in self.samples['obs']['img_nm'][idx]:
                    img_path = os.path.join(self.ped_ori_local_root, 
                                            self.samples['obs']['clip_id']\
                                                [idx][0], 
                                            str(int(float(self.samples['obs']\
                                                          ['obj_id'][idx][0]))), 
                                                          img_nm)
                    ctx_imgs.append(cv2.imread(img_path))
                ctx_imgs = np.stack(ctx_imgs, axis=0)
                # (T, H, W, C) -> (C, T, H, W)
                ctx_imgs = torch.from_numpy(ctx_imgs).float().permute(3, 0, 1, 2)
                # normalize img
                if self.img_norm_mode != 'ori':
                    ctx_imgs = norm_imgs(ctx_imgs, self.img_mean, self.img_std)
                # RGB -> BGR
                if self.target_color_order == 'RGB':
                    ctx_imgs = torch.flip(ctx_imgs, dims=[0])  # 3THW
                # load segs
                ctx_segs = {c:[] for c in self.seg_cls}
                for c in self.seg_cls:
                    for img_nm in self.samples['obs']['img_nm'][idx]:
                        c_id = self.samples['obs']['clip_id'][idx][0]
                        f_nm = img_nm.replace('png', 'pkl')
                        seg_path = os.path.join(self.seg_root, c, c_id, f_nm)
                        with open(seg_path, 'rb') as f:
                            seg = pickle.load(f)
                        ctx_segs[c].append(torch.from_numpy(seg))
                for c in self.seg_cls:
                    ctx_segs[c] = torch.stack(ctx_segs[c], dim=0)  # THW
                # crop seg
                crop_segs = {c:[] for c in self.seg_cls}
                for i in range(ctx_imgs.size(1)):  # T
                    for c in self.seg_cls:
                        crop_seg = crop_local_ctx(
                            torch.unsqueeze(ctx_segs[c][i], dim=0), 
                            obs_bbox_ori[i], 
                            self.ctx_size, 
                            interpo='nearest')  # 1 h w
                        crop_segs[c].append(crop_seg)
                all_seg = []
                for c in self.seg_cls:
                    all_seg.append(torch.stack(crop_segs[c], dim=1))  # 1Thw
                all_seg = torch.stack(all_seg, dim=4)  # 1Thw n_cls
                sample['obs_context'] = all_seg * torch.unsqueeze(ctx_imgs, dim=-1)  # 3 T h w n_cls
                
        if 'sklt' in self.modalities:
            if self.sklt_format == 'pseudo_heatmap':
                cid = str(int(float(self.samples['obs']['clip_id'][idx][0])))
                pid = str(int(float(self.samples['obs']['obj_id'][idx][0])))
                obs_heatmaps = []
                pred_heatmaps = []
                for img_nm in self.samples['obs']['img_nm'][idx]:
                    heatmap_nm = img_nm.replace('.png', '.pkl')
                    heatmap_path = os.path.join(self.sk_p_heatmap_path, cid, pid, heatmap_nm)
                    with open(heatmap_path, 'rb') as f:
                        heatmap = pickle.load(f)
                    obs_heatmaps.append(heatmap)
                for img_nm in self.samples['pred']['img_nm'][idx]:
                    heatmap_nm = img_nm.replace('.png', '.pkl')
                    heatmap_path = os.path.join(self.sk_p_heatmap_path, cid, pid, heatmap_nm)
                    with open(heatmap_path, 'rb') as f:
                        heatmap = pickle.load(f)
                    pred_heatmaps.append(heatmap)
                obs_heatmaps = np.stack(obs_heatmaps, axis=0)  # T C H W
                pred_heatmaps = np.stack(pred_heatmaps, axis=0)  # T C H W
                # T C H W -> C T H W
                obs_skeletons = torch.from_numpy(obs_heatmaps).float().permute(1, 0, 2, 3)  # shape: (17, seq_len, 48, 48)
                pred_skeletons = torch.from_numpy(pred_heatmaps).float().permute(1, 0, 2, 3)  # shape: (17, seq_len, 48, 48)
            elif 'coord' in self.sklt_format:
                cid = str(int(float(self.samples['obs']['clip_id'][idx][0])))
                pid = str(int(float(self.samples['obs']['obj_id'][idx][0])))
                coords = []
                pred_coords = []
                for img_nm in self.samples['obs']['img_nm'][idx]:
                    coord_nm = img_nm.replace('.png', '.pkl')
                    coord_path = os.path.join(self.sk_coord_path, cid, pid, coord_nm)
                    with open(coord_path, 'rb') as f:
                        coord = pickle.load(f)  # nj, 3
                    coords.append(coord[:, :2])  # nj, 2
                for img_nm in self.samples['pred']['img_nm'][idx]:
                    coord_nm = img_nm.replace('.png', '.pkl')
                    coord_path = os.path.join(self.sk_coord_path, cid, pid, coord_nm)
                    with open(coord_path, 'rb') as f:
                        coord = pickle.load(f)  # nj, 3
                    pred_coords.append(coord[:, :2])  # nj, 2(yx)
                coords = np.stack(coords, axis=0)  # T, nj, 2
                pred_coords = np.stack(pred_coords, axis=0)  # T, nj, 2
                obs_skeletons = torch.from_numpy(coords).float().permute(2, 0, 1)  # shape: (2, T, nj)
                pred_skeletons = torch.from_numpy(pred_coords).float().permute(2, 0, 1)  # shape: (2, T, nj)
                # yx -> xy
                obs_skeletons = obs_skeletons.flip(0)
                pred_skeletons = pred_skeletons.flip(0)
                # add offset
                obs_skeletons = sklt_local_to_global(obs_skeletons.float(), obs_bbox_ori.float())
                pred_skeletons = sklt_local_to_global(pred_skeletons.float(), pred_bbox_ori.float())

                if '0-1' in self.sklt_format:
                    obs_skeletons = norm_sklt(obs_skeletons, self.dataset_name)
                    pred_skeletons = norm_sklt(pred_skeletons, self.dataset_name)
                    try:
                        assert torch.max(obs_skeletons) <= 1 and torch.max(pred_skeletons) <= 1, \
                        (torch.max(obs_skeletons), torch.max(pred_skeletons))
                    except:
                        import pdb;pdb.set_trace()
            else:
                raise NotImplementedError(self.sklt_format)
            sample['obs_skeletons'] = obs_skeletons
            sample['pred_skeletons'] = pred_skeletons

        # augmentation
        if self.augment_mode != 'none':
            if self.transforms['random']:
                sample = self._random_augment(sample)
            elif self.transforms['balance']:
                sample['hflip_flag'] = torch.tensor(self.samples[self.obs_or_pred]['hflip_flag'][idx])
                sample = self._augment(sample)

        return sample

    def _add_augment(self, data):
        '''
        data: self.samples, dict of lists(num samples, ...)
        transforms: torchvision.transforms
        '''
        if 'crop' in self.augment_mode:
            if 'img' in self.modalities:
                self.transforms['resized_crop']['img'] = \
                    RandomResizedCrop(size=self.crop_size, # (h, w)
                                    scale=(0.75, 1), 
                                    ratio=(1., 1.))  # w / h
            if 'ctx' in self.modalities:
                self.transforms['resized_crop']['ctx'] = \
                    RandomResizedCrop(size=self.ctx_size, # (h, w)
                                      scale=(0.75, 1), 
                                      ratio=(self.ctx_size[1]/self.ctx_size[0], 
                                             self.ctx_size[1]/self.ctx_size[0]))  # w / h
            if 'sklt' in self.modalities and self.sklt_format == 'pseudo_heatmap':
                self.transforms['resized_crop']['sklt'] = \
                    RandomResizedCrop(size=(48, 48), # (h, w)
                                        scale=(0.75, 1), 
                                        ratio=(1, 1))  # w / h
        if 'hflip' in self.augment_mode:
            if 'random' in self.augment_mode:
                self.transforms['random'] = 1
                self.transforms['balance'] = 0
                self.transforms['hflip'] = RandomHorizontalFlip(p=0.5)
            elif 'balance' in self.augment_mode:
                print(f'Num samples before flip: {self.num_samples}')
                self.transforms['random'] = 0
                self.transforms['balance'] = 1
                imbalance_sets = []

                # init extra samples
                h_flip_samples = {
                    'obs':{},
                    'pred':{}
                }
                for k in data['obs']:
                    h_flip_samples['obs'][k] = []
                    h_flip_samples['pred'][k] = []

                # keys to check
                for k in KEY_2_LABEL:
                    if k in self.augment_mode:
                        imbalance_sets.append(KEY_2_LABEL[k])
                # duplicate samples
                for i in range(len(data['obs']['img_nm'])):
                    for label in imbalance_sets:
                        if data[self.obs_or_pred][label][i][-1] \
                            in LABEL_2_IMBALANCE_CLS[label]:
                            for k in data['obs']:
                                h_flip_samples['obs'][k].append(
                                    copy.deepcopy(data['obs'][k][i]))
                                h_flip_samples['pred'][k].append(
                                    copy.deepcopy(data['pred'][k][i]))
                        break
                h_flip_samples['obs']['hflip_flag'] = \
                    [True for i in range(len(h_flip_samples['obs']['img_nm']))]
                h_flip_samples['pred']['hflip_flag'] = \
                    [True for i in range(len(h_flip_samples['pred']['img_nm']))]
                data['obs']['hflip_flag'] = \
                    [False for i in range(len(data['obs']['img_nm']))]
                data['pred']['hflip_flag'] = \
                    [False for i in range(len(data['pred']['img_nm']))]

                # concat
                for k in data['obs']:
                    data['obs'][k].extend(h_flip_samples['obs'][k])
                    data['pred'][k].extend(h_flip_samples['pred'][k])
            self.num_samples = len(data['obs']['img_nm_int'])
            print(f'Num samples after flip: {self.num_samples}')
        return data

    def _augment(self, sample):
        # flip
        if sample['hflip_flag']:
            if 'img' in self.modalities:
                sample['ped_imgs'] = TVF.hflip(sample['ped_imgs'])
            if 'ctx' in self.modalities:
                sample['obs_context'] = TVF.hflip(sample['obs_context'])
            if 'sklt' in self.modalities and ('heatmap' in self.sklt_format):
                sample['obs_skeletons'] = TVF.hflip(sample['obs_skeletons'])
            if 'traj' in self.modalities:
                sample['obs_bboxes_unnormed'][:, 0], sample['obs_bboxes_unnormed'][:, 2] = \
                    2704 - sample['obs_bboxes_unnormed'][:, 2], 2704 - sample['obs_bboxes_unnormed'][:, 0]
                if '0-1' in self.traj_format:
                    sample['obs_bboxes'][:, 0], sample['obs_bboxes'][:, 2] =\
                         1 - sample['obs_bboxes'][:, 2], 1 - sample['obs_bboxes'][:, 0]
                else:
                    sample['obs_bboxes'][:, 0], sample['obs_bboxes'][:, 2] =\
                         2704 - sample['obs_bboxes'][:, 2], 2704 - sample['obs_bboxes'][:, 0]
            if 'ego' in self.modalities:
                sample['obs_ego'][:, -1] = -sample['obs_ego'][:, -1]
        # resized crop
        if self.transforms['resized_crop']['img'] is not None:
            sample['ped_imgs'], ijhw = self.transforms['resized_crop']['img'](sample['ped_imgs'])
            self.transforms['resized_crop']['img'].randomize_parameters()
            sample['img_ijhw'] = torch.tensor(ijhw)
        if self.transforms['resized_crop']['ctx'] is not None:
            sample['obs_context'], ijhw = self.transforms['resized_crop']['ctx'](sample['obs_context'])
            self.transforms['resized_crop']['ctx'].randomize_parameters()
            sample['ctx_ijhw'] = torch.tensor(ijhw)
        if self.transforms['resized_crop']['sklt'] is not None:
            sample['obs_skeletons'], ijhw = self.transforms['resized_crop']['sklt'](sample['obs_skeletons'])
            self.transforms['resized_crop']['sklt'].randomize_parameters()
            sample['sklt_ijhw'] = torch.tensor(ijhw)
        return sample

    def _random_augment(self, sample):
        # flip
        if self.transforms['hflip'] is not None:
            self.transforms['hflip'].randomize_parameters()
            sample['hflip_flag'] = torch.tensor(self.transforms['hflip'].flag)
            # print('before aug', self.transforms['hflip'].flag, sample['hflip_flag'], self.transforms['hflip'].random_p)
            if 'img' in self.modalities:
                sample['ped_imgs'] = self.transforms['hflip'](sample['ped_imgs'])
            # print('-1', self.transforms['hflip'].flag, sample['hflip_flag'], self.transforms['hflip'].random_p)
            if 'ctx' in self.modalities:
                if self.ctx_format == 'seg_ori_local' or self.ctx_format == 'seg_local':
                    sample['obs_context'] = self.transforms['hflip'](sample['obs_context'].permute(4, 0, 1, 2, 3)).permute(1, 2, 3, 4, 0)
                sample['obs_context'] = self.transforms['hflip'](sample['obs_context'])
            if 'sklt' in self.modalities and ('heatmap' in self.sklt_format):
                sample['obs_skeletons'] = self.transforms['hflip'](sample['obs_skeletons'])
            if 'traj' in self.modalities and self.transforms['hflip'].flag:
                sample['obs_bboxes_unnormed'][:, 0], sample['obs_bboxes_unnormed'][:, 2] = \
                    2704 - sample['obs_bboxes_unnormed'][:, 2], 2704 - sample['obs_bboxes_unnormed'][:, 0]
                if '0-1' in self.traj_format:
                    sample['obs_bboxes'][:, 0], sample['obs_bboxes'][:, 2] =\
                         1 - sample['obs_bboxes'][:, 2], 1 - sample['obs_bboxes'][:, 0]
                else:
                    sample['obs_bboxes'][:, 0], sample['obs_bboxes'][:, 2] =\
                         2704 - sample['obs_bboxes'][:, 2], 2704 - sample['obs_bboxes'][:, 0]
            if 'ego' in self.modalities and self.transforms['hflip'].flag and 'ang' in self.ego_format:
                sample['obs_ego'][:, -1] = -sample['obs_ego'][:, -1]
            
        # resized crop
        if self.transforms['resized_crop']['img'] is not None:
            self.transforms['resized_crop']['img'].randomize_parameters()
            sample['ped_imgs'], ijhw = self.transforms['resized_crop']['img'](sample['ped_imgs'])
            sample['img_ijhw'] = torch.tensor(ijhw)
        if self.transforms['resized_crop']['ctx'] is not None:
            self.transforms['resized_crop']['ctx'].randomize_parameters()
            sample['obs_context'], ijhw = self.transforms['resized_crop']['ctx'](sample['obs_context'])
            sample['ctx_ijhw'] = torch.tensor(ijhw)
        if self.transforms['resized_crop']['sklt'] is not None:
            self.transforms['resized_crop']['sklt'].randomize_parameters()
            sample['obs_skeletons'], ijhw = self.transforms['resized_crop']['sklt'](sample['obs_skeletons'])
            sample['sklt_ijhw'] = torch.tensor(ijhw)
        return sample

    def add_cid(self):
        annos = {}
        ids = {}
        n_clip = 0
        for cid in self.clip_id_list:
            n_clip += 1
            ids[cid] = {'pid':set(), 'vid':set()}
            csv_path = os.path.join(self.ori_root, 'clip_'+cid+'.csv')
            # print(f'cid {cid} n clips {n_clip}')
            clip_obj_info = self.read_obj_csv(csv_path)
            for i in range(len(clip_obj_info)):
                line = clip_obj_info[i]
                assert len(line) == 16
                if line[1] == 'person':
                    ids[cid]['pid'].add((cid, line[2]))
                else:
                    ids[cid]['vid'].add((cid, line[2]))
                clip_obj_info[i].append(cid)
            annos[cid] = self.str2ndarray(clip_obj_info)

        return annos, ids

    def get_p_tracks(self, annos):
        p_tracks = {'clip_id': [],
                    'img_nm': [],  # str
                    'img_nm_int': [],
                    'obj_id': [],
                    'bbox': [],
                    # 'motion_status': [],
                    'communicative': [],
                    'complex_context': [],
                    'atomic_actions': [],
                    'simple_context': [],
                    'transporting': [],
                    'age': [],
                    'ego_motion': []}
        for cid in self.ids.keys():
            # load ego motion
            ego_v_path = os.path.join(self.ori_root, 'clip_'+cid, 'synced_sensors.csv')
            ego_v_info = self.read_ego_csv(ego_v_path)  # dict {'img_nm': [accel, yaw]}
            clip_annos = annos[cid]
            for _, pid in self.ids[cid]['pid']:
                # init new track
                for k in p_tracks.keys():
                    p_tracks[k].append([])
                # filter lines
                lines = clip_annos[(clip_annos[:, 2] == pid) & (clip_annos[:, 1] == 'person')]
                for line in lines:
                    # check if required labels exist
                    flg = 0
                    for label in LABEL2DICT:
                        idx = LABEL2COLUMN[label]
                        # print(line[idx], type(line[idx]))
                        cur_s = line[idx]
                        if cur_s == '':
                            flg = 1
                            OCC_NUM += 1
                            print('occlusion', OCC_NUM)
                            break
                        elif (label in LABEL2DICT) and \
                            (cur_s not in LABEL2DICT[label]):
                            flg = 1
                            # print('Class not to recognize: ', line[idx])
                            break
                    if flg == 1:
                        # pop current track
                        if self.pop_occl_track:
                            for k in p_tracks:
                                p_tracks[k].pop(-1)
                            break
                        else:
                            # init a new track
                            for k in p_tracks.keys():
                                p_tracks[k].append([])
                            continue
                    cur_img_nm_int = int(line[0].replace('.png', ''))
                    # check continuity
                    if len(p_tracks['img_nm_int'][-1])>0 and \
                        cur_img_nm_int-p_tracks['img_nm_int'][-1][-1]>6:
                        # init a new track
                        for k in p_tracks.keys():
                            p_tracks[k].append([])
                    p_tracks['clip_id'][-1].append(cid)  # str
                    p_tracks['obj_id'][-1].append(str(int(float(pid))))  # str
                    p_tracks['img_nm'][-1].append(line[0])  # str
                    p_tracks['img_nm_int'][-1].append(cur_img_nm_int)
                    tlhw = list(map(float, line[3: 7]))
                    ltrb = [tlhw[1], tlhw[0], tlhw[1]+tlhw[3], tlhw[0]+tlhw[2]]
                    p_tracks['bbox'][-1].append(ltrb)
                    p_tracks['communicative'][-1].append(COMMUNICATIVE_LABEL[line[10]])
                    p_tracks['complex_context'][-1].append(COMPLEX_CONTEXTUAL_LABEL[line[11]])
                    p_tracks['atomic_actions'][-1].append(ATOM_ACTION_LABEL[line[12]])
                    p_tracks['simple_context'][-1].append(SIMPLE_CONTEXTUAL_LABEL[line[13]])
                    p_tracks['transporting'][-1].append(TRANSPORTIVE_LABEL[line[14]])
                    p_tracks['age'][-1].append(AGE_LABEL[line[15]])
                    ego_motion = ego_v_info[line[0].replace('.png', '')]  # 
                    p_tracks['ego_motion'][-1].append(list(map(float, ego_motion)))
        
        num_tracks = len(p_tracks['clip_id'])
        for k in p_tracks.keys():
            assert len(p_tracks[k]) == num_tracks, (k, len(p_tracks[k]), num_tracks)

        return p_tracks, num_tracks

    def get_v_tracks(self, annos):  # TBD
        v_tracks = {'clip_id': [],
                    'img_nm': [],
                    'img_nm_int': [],
                    'obj_type': [],
                    'obj_id': [],
                    'bbox': [],
                    'motion_status': [],
                    'trunk_open': [],
                    'doors_open': [],
                    'ego_motion': []}
        for cid in self.ids.keys():
            # load ego motion
            ego_v_path = os.path.join(self.ori_root, 'clip_'+cid, 'synced_sensors.csv')
            ego_v_info = self.read_ego_csv(ego_v_path)  # dict {'img_nm': [info]}
            clip_annos = annos[cid]
            for _, vid in self.ids[cid]['vid']:
                # init new track
                for k in v_tracks.keys():
                    v_tracks[k].append([])
                # filter lines
                lines = clip_annos[(clip_annos[:, 2] == vid) & (clip_annos[:, 1] != 'person')]
                for line in lines:
                    v_tracks['clip_id'][-1].append(cid)
                    v_tracks['obj_id'][-1].append(str(int(float(vid))))
                    v_tracks['img_nm'][-1].append(line[0])
                    v_tracks['img_nm_int'][-1].append(int(line[0].replace('.png', '')))
                    v_tracks['obj_type'][-1].append(line[1])
                    tlhw = list(map(float, line[3: 7]))
                    ltrb = [tlhw[1], tlhw[0], tlhw[1]+tlhw[3], tlhw[0]+tlhw[2]]
                    v_tracks['bbox'][-1].append(ltrb)
                    v_tracks['motion_status'][-1].append(MOTOIN_STATUS_LABEL[line[8]])
                    v_tracks['trunk_open'][-1].append(line[7])
                    v_tracks['doors_open'][-1].append(line[9])
                    ego_motion = ego_v_info[line[0].replace('.png', '')]
                    v_tracks['ego_motion'][-1].append(list(map(float, ego_motion)))

        num_tracks = len(v_tracks['clip_id'])
        for k in v_tracks.keys():
            assert len(v_tracks[k]) == num_tracks, (k, len(v_tracks[k]), num_tracks)

        return v_tracks, num_tracks

    def track2sample(self, tracks):
        seq_len = self._obs_len + self._pred_len
        overlap_s = self._obs_len if self.overlap_ratio == 0 \
            else int((1 - self.overlap_ratio) * self._obs_len)
        overlap_s = 1 if overlap_s < 1 else overlap_s
        samples = {}
        for dt in tracks.keys():
            try:
                samples[dt] = tracks[dt]
            except KeyError:
                raise ('Wrong data type is selected %s' % dt)

        # split tracks to fixed length samples
        print('---------------Split tracks to samples---------------')
        print(samples.keys())
        for k in tqdm(samples.keys()):
            _samples = []
            for track in samples[k]:
                if self.tte is not None:
                    start_idx = len(track) - seq_len - self.tte[1]
                    end_idx = len(track) - seq_len - self.tte[0]
                    _samples.extend(
                        [track[i:i + seq_len] for i in range(start_idx, 
                                                             end_idx + 1, 
                                                             overlap_s)])
                else:
                    _samples.extend(
                        [track[i: i+seq_len] for i in range(0, 
                                                             len(track) - seq_len + 1, 
                                                             overlap_s)])
            samples[k] = _samples

        #  Normalize tracks by subtracting bbox/center at first time step from the rest
        print('---------------Normalize traj---------------')
        bbox_normed = copy.deepcopy(samples['bbox'])
        if self.offset_traj:
            for i in range(len(bbox_normed)):
                bbox_normed[i] = np.subtract(bbox_normed[i][:], bbox_normed[i][0]).tolist()
        samples['bbox_normed'] = bbox_normed

        # split obs and pred
        print('---------------Split obs and pred---------------')
        obs_slices = {}
        pred_slices = {}
        for k in samples.keys():
            obs_slices[k] = []
            pred_slices[k] = []
            obs_slices[k].extend([d[0:self._obs_len] for d in samples[k]])
            pred_slices[k].extend([d[self._obs_len:] for d in samples[k]])

        all_samples = {
            'obs': obs_slices,
            'pred': pred_slices
        }

        return all_samples

    def get_imgnm_to_objid(self, 
                           p_tracks, 
                           v_tracks, 
                           save_path):
        # cid_to_imgnm_to_oid_to_info: cid -> img name -> obj type (ped/veh) -> obj id -> bbox/ego motion
        imgnm_to_oid_to_info = {}
        n_p_tracks = len(p_tracks['bbox'])
        print(f'Saving img_nm to obj_id dict of {self.dataset_name}')
        print('Pedestrian')
        for i in range(n_p_tracks):
            cid = p_tracks['clip_id'][i][0]  # str
            oid = p_tracks['obj_id'][i][0]  # str
            if cid not in imgnm_to_oid_to_info:
                imgnm_to_oid_to_info[cid] = {}
            for j in range(len(p_tracks['img_nm'][i])):
                imgnm = p_tracks['img_nm'][i][j]
                # initialize the dict of the img
                if imgnm not in imgnm_to_oid_to_info[cid]:
                    imgnm_to_oid_to_info[cid][imgnm] = {}
                    imgnm_to_oid_to_info[cid][imgnm]['ped'] = {}
                    imgnm_to_oid_to_info[cid][imgnm]['veh'] = {}
                # initialize the dict of the obj
                bbox = p_tracks['bbox'][i][j]
                imgnm_to_oid_to_info[cid][imgnm]['ped'][oid] = {}
                imgnm_to_oid_to_info[cid][imgnm]['ped'][oid]['bbox'] = bbox
        print('Vehicle')
        n_v_tracks = len(v_tracks['bbox'])
        for i in range(n_v_tracks):
            cid = v_tracks['clip_id'][i][0]
            oid = v_tracks['obj_id'][i][0]
            if cid not in imgnm_to_oid_to_info:
                imgnm_to_oid_to_info[cid] = {}
            for j in range(len(v_tracks['img_nm'][i])):
                imgnm = v_tracks['img_nm'][i][j]
                # initialize the dict of the img
                if imgnm not in imgnm_to_oid_to_info[cid]:
                    imgnm_to_oid_to_info[cid][imgnm] = {}
                    imgnm_to_oid_to_info[cid][imgnm]['ped'] = {}
                    imgnm_to_oid_to_info[cid][imgnm]['veh'] = {}
                # initialize the dict of the obj
                bbox = v_tracks['bbox'][i][j]
                imgnm_to_oid_to_info[cid][imgnm]['veh'][oid] = {}
                imgnm_to_oid_to_info[cid][imgnm]['veh'][oid]['bbox'] = bbox
        
        with open(save_path, 'wb') as f:
            pickle.dump(imgnm_to_oid_to_info, f)
        
        return imgnm_to_oid_to_info

    def filter_short_tracks(self, tracks, min_len):
        '''
        tracks: dict
        '''
        idx = []
        _tracks = copy.deepcopy(tracks)
        n_tracks = len(_tracks['img_nm'])
        for i in range(n_tracks):
            if len(_tracks['img_nm'][i]) < min_len:
                idx.append(i)
        # print('short track to remove',len(idx), idx)
        # for i in idx:
        #     print(_tracks['clip_id'][i], _tracks['obj_id'][i])
        for i in reversed(idx):
            for k in _tracks.keys():
                _tracks[k].pop(i)
        
        return _tracks, len(_tracks['img_nm'])

    def multi2binary(self, labels, idxs):
        '''
        labels: list (n_samples, seq_len)
        idxs: list (int,...)
        '''
        bi_labels = []
        for sample in labels:
            bi_labels.append([])
            for t in sample:
                if t in idxs:
                    bi_labels[-1].append(1)
                else:
                    bi_labels[-1].append(0)
        return bi_labels

    def get_neighbor_relation(self,
                      samples,
                      save_path,
                      padding_val=0):
        if os.path.exists(save_path) and False:
            with open(save_path, 'rb') as f:
                neighbor_seq = pickle.load(f)
        else:
            n_sample = len(samples['obs']['bbox'])
            relations = []
            neighbor_bbox = []
            neighbor_oid = []
            neighbor_cls = []
            print('Getting neighbor sequences')
            for i in tqdm(range(n_sample)):
                target_cid = samples['obs']['clip_id'][i][0]  # str
                target_oid = samples['obs']['obj_id'][i][0]  # str
                target_bbox_seq = np.array(samples['obs']['bbox'][i])  # T, 4
                obslen = len(target_bbox_seq)
                bbox_seq_dict = {'ped':{},
                                'veh':{}}
                for j in range(obslen):
                    imgnm = samples['obs']['img_nm'][i][j]  # str
                    try:
                        cur_ped_ids = set(self.imgnm_to_objid[target_cid][imgnm]['ped'].keys())
                    except:
                        import pdb;pdb.set_trace()
                    cur_ped_ids.remove(target_oid)
                    cur_veh_ids = set(self.imgnm_to_objid[target_cid][imgnm]['veh'].keys())
                    # ped neighbor for cur sample
                    # existing neighbor
                    for oid in set(bbox_seq_dict['ped'].keys())&cur_ped_ids:
                        bbox = np.array(self.imgnm_to_objid[target_cid][imgnm]['ped'][oid]['bbox'])
                        bbox_seq_dict['ped'][oid].append(bbox)
                    # first appearing neighbor
                    for oid in cur_ped_ids-set(bbox_seq_dict['ped'].keys()):
                        bbox = np.array(self.imgnm_to_objid[target_cid][imgnm]['ped'][oid]['bbox'])
                        bbox_seq_dict['ped'][oid] = [np.ones([4])*padding_val]*j + [bbox]  # T, 4
                    # disappeared neighbor
                    for oid in set(bbox_seq_dict['ped'].keys())-cur_ped_ids:
                        bbox_seq_dict['ped'][oid].append(np.ones([4])*padding_val)
                    
                    # veh neighbor for cur sample
                    # existing neighbor
                    for oid in set(bbox_seq_dict['veh'].keys())&cur_veh_ids:
                        bbox = np.array(self.imgnm_to_objid[target_cid][imgnm]['veh'][oid]['bbox'])
                        bbox_seq_dict['veh'][oid].append(bbox)
                    # first appearing neighbor
                    for oid in cur_veh_ids-set(bbox_seq_dict['veh'].keys()):
                        bbox = np.array(self.imgnm_to_objid[target_cid][imgnm]['veh'][oid]['bbox'])
                        bbox_seq_dict['veh'][oid] = [np.ones([4])*padding_val]*j + [bbox]  # T, 4
                    # disappeared neighbor
                    for oid in set(bbox_seq_dict['veh'].keys())-cur_veh_ids:
                        bbox_seq_dict['veh'][oid].append(np.ones([4])*padding_val)
                cur_neighbor_bbox = []
                cur_neighbor_oid = []
                cur_neighbor_cls = []
                for oid in bbox_seq_dict['ped']:
                    # import pdb;pdb.set_trace()
                    if oid == target_oid:
                        import pdb;pdb.set_trace()
                        raise ValueError()
                    cur_neighbor_bbox.append(bbox_seq_dict['ped'][oid])  # T, 4
                    cur_neighbor_oid.append(int(oid))
                    cur_neighbor_cls.append(0)
                for oid in bbox_seq_dict['veh']:
                    cur_neighbor_bbox.append(bbox_seq_dict['veh'][oid])  # T, 4
                    cur_neighbor_oid.append(int(oid))
                    cur_neighbor_cls.append(1)
                if len(cur_neighbor_bbox) == 0:
                    cur_neighbor_bbox = np.zeros([1, obslen, 4])
                    cur_neighbor_oid = np.ones([1]) * (-1)
                    cur_neighbor_cls = np.zeros([1])
                else:
                    cur_neighbor_bbox = np.array(cur_neighbor_bbox)  # K T 4
                    cur_neighbor_oid = np.array(cur_neighbor_oid)  # K,
                    cur_neighbor_cls = np.array(cur_neighbor_cls)  # K,
                cur_relations = bbox2d_relation_multi_seq(target_bbox_seq,
                                                        cur_neighbor_bbox,
                                                        rela_func='log_bbox_reg')  # K T 4
                # concat cls label to relations
                cur_relations = np.concatenate([cur_relations, 
                                                cur_neighbor_cls.reshape([-1, 1, 1]).repeat(obslen, axis=1)],
                                                -1)
                relations.append(cur_relations.transpose(1,0,2))  # K T 5 -> T K 5
                neighbor_bbox.append(cur_neighbor_bbox.transpose(1,0,2))  # K T 4 -> T K 4
                
                neighbor_oid.append(cur_neighbor_oid)  # K
                neighbor_cls.append(cur_neighbor_cls)  # K
            neighbor_seq = {}
            neighbor_seq['neighbor_relations'] = relations
            neighbor_seq['neighbor_bbox'] = neighbor_bbox
            neighbor_seq['neighbor_oid'] = neighbor_oid
            neighbor_seq['neighbor_cls'] = neighbor_cls
            # import pdb;pdb.set_trace()
            with open(save_path, 'wb') as f:
                pickle.dump(neighbor_seq, f)
        # add neighbor info to samples
        for k in neighbor_seq:
            samples['obs'][k] = neighbor_seq[k]
            # print(f'num of {k}: {len(neighbor_seq[k])}')
        # print(len(samples['obs']['obj_id']))
        return samples
    
    def str2ndarray(self, anno_list):
        return np.array(anno_list)

    def read_obj_csv(self, csv_path):
        res = []
        with open (csv_path, 'r') as f:
            reader = csv.reader(f)
            for item in reader:
                if reader.line_num == 1:
                    continue
                res.append(item)
        
        return res
    
    def read_ego_csv(self, csv_path):
        res = {}
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                img_nm = line[1].split('/')[-1].replace('.png', '')
                res[img_nm] = [line[3], line[5]]  # accel, angle vel
            
        return res

    def downsample_seq(self):
        for k in self.samples['obs']:
            # if 'neighbor' in k:
            #     import pdb;pdb.set_trace()
            if len(self.samples['obs'][k][0]) == self._obs_len and (k not in ('neighbor_oid', 'neighbor_cls')):
                new_k = []
                for s in range(len(self.samples['obs'][k])):
                    ori_seq = self.samples['obs'][k][s]
                    new_seq = []
                    for i in range(0, self._obs_len, self.seq_interval+1):
                        new_seq.append(ori_seq[i])
                    new_k.append(np.array(new_seq))
                    assert len(new_k[s]) == self.obs_len, (k, len(new_k), self.obs_len)
                # new_k = np.array(new_k)
                self.samples['obs'][k] = new_k
        for k in self.samples['pred']:
            if len(self.samples['pred'][k][0]) == self._pred_len:
                new_k = []
                for s in range(len(self.samples['pred'][k])):
                    ori_seq = self.samples['pred'][k][s]
                    new_seq = []
                    for i in range(0, self._pred_len, self.seq_interval+1):
                        new_seq.append(ori_seq[i])
                    new_k.append(np.array(new_seq))
                    assert len(new_k[s]) == self.pred_len, (k, len(new_k), self.pred_len)
                # new_k = np.array(new_k)
                self.samples['pred'][k] = new_k

# def check_labels():
#     not_matched = set()
#     ori_data_root = '/home/y_feng/workspace6/datasets/TITAN/honda_titan_dataset/dataset'
#     for d in os.listdir(ori_data_root):
#         if 'clip_' in d and '.csv' in d:
#             csv_path = os.path.join(ori_data_root, d)
#             with open(csv_path, 'r') as f:
#                 reader = csv.reader(f)
#                 for line in reader:
#                     if reader.line_num == 1:
#                         continue
#                     if line[8] not in MOTOIN_STATUS_LABEL.keys():
#                         not_matched.add(line[8])
#                     if line[10] not in COMMUNICATIVE_LABEL.keys():
#                         not_matched.add(line[10])
#                     if line[11] not in COMPLEX_CONTEXTUAL_LABEL.keys():
#                         not_matched.add(line[11])
#                     if line[12] not in ATOM_ACTION_LABEL.keys():
#                         not_matched.add(line[12])
#                     if line[13] not in SIMPLE_CONTEXTUAL_LABEL.keys():
#                         not_matched.add(line[13])
#                     if line[14] not in TRANSPORTIVE_LABEL.keys():
#                         not_matched.add(line[14])
#                     if line[15] not in AGE_LABEL.keys():
#                         not_matched.add(line[15])
#             print(d, ' done')
#     print(not_matched)

# def crop_imgs(tracks, resize_mode='even_padded', target_size=(224, 224), obj_type='p'):
#     crop_root = '/home/y_feng/workspace6/datasets/TITAN/TITAN_extra/cropped_images'
#     makedir(crop_root)
#     data_root = '/home/y_feng/workspace6/datasets/TITAN/honda_titan_dataset/dataset'
#     if obj_type == 'p':
#         crop_obj_path = os.path.join(crop_root, resize_mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h', 'ped')
#         makedir(crop_obj_path)
#     else:
#         crop_obj_path = os.path.join(crop_root, resize_mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h', 'veh')
#         makedir(crop_obj_path)
#     for i in tqdm(range(len(tracks['clip_id']))):
#         cid = int(tracks['clip_id'][i][0])
#         oid = int(float(tracks['obj_id'][i][0]))
#         cur_clip_path = os.path.join(crop_obj_path, str(cid))
#         makedir(cur_clip_path)
#         cur_obj_path = os.path.join(cur_clip_path, str(oid))
#         makedir(cur_obj_path)
        
#         for j in range(len(tracks['clip_id'][i])):
#             img_nm = tracks['img_nm'][i][j]
#             l, t, r, b = list(map(int, tracks['bbox'][i][j]))
#             img_path = os.path.join(data_root, 'images_anonymized', 'clip_'+str(cid), 'images', img_nm)
#             tgt_path = os.path.join(cur_obj_path, img_nm)
#             img = cv2.imread(img_path)
#             cropped = img[t:b, l:r]
#             if resize_mode == 'ori':
#                 resized = cropped
#             elif resize_mode == 'resized':
#                 resized = cv2.resize(cropped, target_size)
#             elif resize_mode == 'even_padded':
#                 h = b-t
#                 w = r-l
#                 if  float(w) / h > float(target_size[0]) / target_size[1]:
#                     ratio = float(target_size[0]) / w
#                 else:
#                     ratio = float(target_size[1]) / h
#                 new_size = (int(w*ratio), int(h*ratio))
#                 cropped = cv2.resize(cropped, new_size)
#                 w_pad = target_size[0] - new_size[0]
#                 h_pad = target_size[1] - new_size[1]
#                 l_pad = w_pad // 2
#                 r_pad = w_pad - l_pad
#                 t_pad = h_pad // 2
#                 b_pad = h_pad - t_pad
#                 resized = cv2.copyMakeBorder(cropped,t_pad,b_pad,l_pad,r_pad,cv2.BORDER_CONSTANT,value=(0, 0, 0))  # t, b, l, r
#                 assert (resized.shape[1], resized.shape[0]) == target_size
#             else:
#                 raise NotImplementedError(resize_mode)
#             cv2.imwrite(tgt_path, resized)
#         print(i, cid, cur_obj_path, 'done')

# def save_context_imgs(tracks, mode='local', target_size=(224, 224), obj_type='p'):
#     ori_H, ori_W = 1520, 2704
#     crop_root = '/home/y_feng/workspace6/datasets/TITAN/TITAN_extra/context'
#     makedir(crop_root)
#     data_root = '/home/y_feng/workspace6/datasets/TITAN/honda_titan_dataset/dataset'
#     if obj_type == 'p':
#         crop_obj_path = os.path.join(crop_root, mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h', 'ped')
#         makedir(crop_obj_path)
#     else:
#         crop_obj_path = os.path.join(crop_root, mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h', 'veh')
#         makedir(crop_obj_path)
    
#     if mode == 'local':
#         for i in range(len(tracks['clip_id'])):  # tracks
#             cid = int(tracks['clip_id'][i][0])
#             oid = int(float(tracks['obj_id'][i][0]))
#             cur_clip_path = os.path.join(crop_obj_path, str(cid))
#             makedir(cur_clip_path)
#             cur_obj_path = os.path.join(cur_clip_path, str(oid))
#             makedir(cur_obj_path)
#             for j in range(len(tracks['clip_id'][i])):  # time steps in each track
#                 img_nm = tracks['img_nm'][i][j]
#                 l, t, r, b = list(map(int, tracks['bbox'][i][j]))
#                 img_path = os.path.join(data_root, 'images_anonymized', 'clip_'+str(cid), 'images', img_nm)
#                 tgt_path = os.path.join(cur_obj_path, img_nm)
#                 img = cv2.imread(img_path)
#                 # mask target pedestrian
#                 rect = np.array([[l, t], [r, t], [r, b], [l, b]])
#                 masked = cv2.fillConvexPoly(img, rect, (127, 127, 127))
#                 # crop local context
#                 x = (l+r) // 2
#                 y = (t+b) // 2
#                 h = b-t
#                 w = r-l
#                 crop_h = h*2
#                 crop_w = h*2
#                 crop_l = max(x-h, 0)
#                 crop_r = min(x+h, ori_W)
#                 crop_t = max(y-h, 0)
#                 crop_b = min(y+h, ori_W)
#                 cropped = masked[crop_t:crop_b, crop_l:crop_r]
#                 l_pad = max(h-x, 0)
#                 r_pad = max(x+h-ori_W, 0)
#                 t_pad = max(h-y, 0)
#                 b_pad = max(y+h-ori_H, 0)
#                 cropped = cv2.copyMakeBorder(cropped, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, value=(127, 127, 127))
#                 assert cropped.shape[0] == crop_h and cropped.shape[1] == crop_w, (cropped.shape, (crop_h, crop_w))
#                 resized = cv2.resize(cropped, target_size)
#                 cv2.imwrite(tgt_path, resized)
#             print(i, cid, oid, cur_obj_path, 'done')
#     elif mode == 'ori_local':
#         for i in range(len(tracks['clip_id'])):  # tracks
#             cid = int(tracks['clip_id'][i][0])
#             oid = int(float(tracks['obj_id'][i][0]))
#             cur_clip_path = os.path.join(crop_obj_path, str(cid))
#             makedir(cur_clip_path)
#             cur_obj_path = os.path.join(cur_clip_path, str(oid))
#             makedir(cur_obj_path)
#             for j in range(len(tracks['clip_id'][i])):  # time steps in each track
#                 img_nm = tracks['img_nm'][i][j]
#                 l, t, r, b = list(map(int, tracks['bbox'][i][j]))
#                 img_path = os.path.join(data_root, 'images_anonymized', 'clip_'+str(cid), 'images', img_nm)
#                 tgt_path = os.path.join(cur_obj_path, img_nm)
#                 img = cv2.imread(img_path)
#                 # crop local context
#                 x = (l+r) // 2
#                 y = (t+b) // 2
#                 h = b-t
#                 w = r-l
#                 crop_h = h*2
#                 crop_w = h*2
#                 crop_l = max(x-h, 0)
#                 crop_r = min(x+h, ori_W)
#                 crop_t = max(y-h, 0)
#                 crop_b = min(y+h, ori_W)
#                 cropped = img[crop_t:crop_b, crop_l:crop_r]
#                 l_pad = max(h-x, 0)
#                 r_pad = max(x+h-ori_W, 0)
#                 t_pad = max(h-y, 0)
#                 b_pad = max(y+h-ori_H, 0)
#                 cropped = cv2.copyMakeBorder(cropped, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, value=(127, 127, 127))
#                 assert cropped.shape[0] == crop_h and cropped.shape[1] == crop_w, (cropped.shape, (crop_h, crop_w))
#                 resized = cv2.resize(cropped, target_size)
#                 cv2.imwrite(tgt_path, resized)
#             print(i, cid, oid, cur_obj_path, 'done')
#     else:
#         raise NotImplementedError(mode)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='crop')

    parser.add_argument('--h', type=int, default=224)
    parser.add_argument('--w', type=int, default=224)
    # crop args
    parser.add_argument('--resize_mode', type=str, default='even_padded')
    parser.add_argument('--subset', type=str, default='train')
    # context args
    parser.add_argument('--ctx_mode', type=str, default='local')
    args = parser.parse_args()

    