from hashlib import new
import pickle
from re import T
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.transforms import functional as TVF
import numpy as np
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import time
import copy
import tqdm
import os
from tqdm import tqdm
import pdb

from .pie_data import PIE
from .jaad_data import JAAD
from ..utils import mapping_20, makedir, ltrb2xywh, coord2pseudo_heatmap, cls_weights
from ..data.transforms import RandomHorizontalFlip, RandomResizedCrop, crop_local_ctx
from ..data.normalize import img_mean_std_BGR, norm_imgs, sklt_local_to_global, norm_bbox, norm_sklt
from ..general import HiddenPrints
from ..data.bbox import bbox2d_relation_multi_seq, pad_neighbor
from .dataset_id import DATASET_TO_ID, ID_TO_DATASET
from config import dataset_root


# PIE: set id --> video id --> img id
#                          --> ped id
# JAAD: video id --> img id
#                --> ped id
class PIEDataset(Dataset):
    def __init__(self, 
                 dataset_name='PIE', 
                 offset_traj=False, 
                 img_norm_mode='torch', 
                 data_split_type='default',
                 seq_type='crossing', 
                 obs_len=4, pred_len=4, overlap_ratio=0.5,
                 obs_fps=2,
                 do_balance=False,
                 subset='train', 
                 bbox_size=(224, 224), ctx_size=(224, 224), 
                 target_color_order='BGR', crop_from_ori=False,
                 resize_mode='even_padded', min_wh=None, max_occ=2, 
                 modalities=['img', 'sklt', 'ctx', 'traj', 'ego', 'social'],
                 img_format='',
                 sklt_format='coord', 
                 ctx_format='ori_local', 
                 traj_format='ltrb', 
                 ego_format='accel',
                 social_format='rel_loc',
                 pred_img=0,
                 pred_context=0, pred_context_mode='ori',
                 small_set=0,
                 tte=[30, 60],
                 recog_act=0,
                 augment_mode='random_hflip',
                 seg_cls=['person', 'vehicle', 'road', 'traffic_light'],
                 speed_unit='m/s',
                 max_n_neighbor=10,
                 ):
        super(Dataset, self).__init__()
        self.dataset_name = dataset_name
        print(f'------------------Init{self.dataset_name}------------------')
        self.fps = 30
        self.img_size = (1080, 1920)
        self.seq_type = seq_type
        self.tte = tte
        self.recog_act = recog_act
        self.subset = subset
        self.offset_traj = offset_traj
        self.img_norm_mode = img_norm_mode
        self.target_color_order = target_color_order
        self.img_mean, self.img_std = img_mean_std_BGR(self.img_norm_mode)  # BGR
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.overlap_ratio = overlap_ratio
        self.bbox_size = bbox_size
        self.seq_interval = self.fps // obs_fps - 1

        # interval
        self._obs_len = self.obs_len * (self.seq_interval + 1)
        self._pred_len = self.pred_len * (self.seq_interval + 1)

        self.crop_from_ori = crop_from_ori
        self.resize_mode = resize_mode
        self.modalities = modalities
        self.img_format = img_format
        self.sklt_format = sklt_format
        self.ctx_format = ctx_format
        self.ctx_size = ctx_size
        self.traj_format = traj_format
        self.ego_format = ego_format
        self.social_format = social_format
        # self.seg_class_idx = [11, 13, 6, 7]  # person, car, light, sign
        self.pred_img = pred_img
        self.pred_context = pred_context
        self.pred_context_mode = pred_context_mode
        self.augment_mode = augment_mode
        self.seg_cls = seg_cls
        self.max_n_neighbor = max_n_neighbor
        self.ego_motion_key = 'ego_accel' if 'accel' in self.ego_format else 'obd_speed'
        if self.dataset_name == 'JAAD':
            self.ego_motion_key = 'vehicle_act'
        self.speed_unit = speed_unit
        self.transforms = {'random': 0,
                            'hflip': None,
                            'resized_crop': {'img': None,
                                            'ctx': None,
                                            'sklt': None}}

        # data opts
        if dataset_name == 'PIE':
            self.root_path = os.path.join(dataset_root, 'PIE_dataset')
            self.data_opts = {'normalize_bbox': offset_traj,
                         'fstride': 1,
                         'sample_type': 'all',
                         'height_rng': [0, float('inf')],
                         'squarify_ratio': 0,
                         # kfold, random, default. default: set03 for test
                         'data_split_type': data_split_type,  
                         # crossing , intention
                         'seq_type': seq_type,  
                         'min_track_size': 1,  # discard tracks that are shorter
                         'max_size_observe': 1,  # number of observation frames
                         'max_size_predict': 1,  # number of prediction frames
                         'seq_overlap_rate': overlap_ratio,  # how much consecutive sequences overlap
                         'balance': do_balance,  # balance the training and testing samples
                         'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
                         'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
                         'seq_type': seq_type,
                         'encoder_input_type': ['bbox', 'obd_speed'],
                         'decoder_input_type': [],
                         'output_type': ['intention_binary', 'bbox']
                         }
            with HiddenPrints():
                data_base = PIE(data_path=self.root_path)
        else:
            self.root_path = os.path.join(dataset_root, 'JAAD')
            self.data_opts = {'fstride': 1,
                            'sample_type': 'all',  
                            'subset': 'default',
                            'data_split_type': data_split_type,
                            'seq_type': seq_type,
                            # 'min_track_size': self._obs_len + self._pred_len,  # discard tracks that are shorter
                            'min_track_size': 1,
                            'height_rng': [0, float('inf')],
                            'squarify_ratio': 0,
                            'random_params': {'ratios': None,
                                            'val_data': True,
                                            'regen_data': True},
                            'kfold_params': {'num_folds': 5, 'fold': 1}}
            with HiddenPrints():
                data_base = JAAD(data_path=self.root_path)
        self.extra_root = self.root_path
        self.sk_vis_path = os.path.join(self.root_path, 'sk_vis/even_padded/288w_by_384h/')
        self.sk_coord_path = os.path.join(self.root_path, 'sk_coords/even_padded/288w_by_384h/')
        self.sk_heatmap_path = os.path.join(self.root_path, 'sk_heatmaps/even_padded/288w_by_384h/')
        self.sk_p_heatmap_path = os.path.join(self.root_path, 'sk_pseudo_heatmaps/even_padded/48w_by_48h/')
        self.sam_seg_root = os.path.join(self.root_path, 'seg_sam')
        self.veh_info_path = os.path.join(self.root_path, 'veh_tracks.pkl')
        self.imgnm_to_objid_path = os.path.join(self.root_path, 
                                                'imgnm_to_objid_to_ann.pkl')
        
        if self.tte is not None:
            self.data_opts['min_track_size'] += self.tte[1]

        self.cropped_path = os.path.join(self.root_path, 
                                         'cropped_images', 
                                         resize_mode, 
                                         str(bbox_size[0])+'w_by_'\
                                            +str(bbox_size[1])+'h')
        
        # get pedestrian tracks
        # all data: {'image', 'ped_id', 'bbox', 'center', 'occlusion', 'vehicle_act', 'intention_binary', 'activities', 'image_dimension'}
        with HiddenPrints():
            self.p_tracks = \
                data_base.generate_data_trajectory_sequence(image_set=self.subset, 
                                                            **self.data_opts)
        # remove the image dimension key
        self.p_tracks.pop('image_dimension', None)
        # remove 720x1280 videos
        if self.dataset_name == 'JAAD':
            self.p_tracks = self.rm_720x1280(self.p_tracks)
        
        # convert speed unit
        if self.speed_unit == 'm/s':
            if self.dataset_name == 'JAAD' and 'ego' in self.modalities:
                raise ValueError('Cannot turn ego motion in JAAD into m/s.')
            elif self.dataset_name == 'PIE':
                self.p_tracks['obd_speed'] = \
                    self._convert_speed_unit(self.p_tracks['obd_speed'])
        # calc acceleration
        if self.dataset_name == 'PIE':
            if 'accel' in self.ego_format:
                self.p_tracks = self._get_accel(self.p_tracks)

        # get vehicle tracks
        with open(self.veh_info_path, 'rb') as f:
            self.v_tracks = pickle.load(f)
        
        # get img name to obj id dict
        if self.dataset_name == 'PIE':
            if not os.path.exists(self.imgnm_to_objid_path) or True:
                # (set id ->) vid id -> img nm -> ped/veh -> oid -> bbox
                self.imgnm_to_objid = \
                    self.get_imgnm_to_objid(self.p_tracks, 
                                            self.v_tracks, 
                                            self.imgnm_to_objid_path)
            else:
                with open(self.imgnm_to_objid_path, 'rb') as f:
                    self.imgnm_to_objid = pickle.load(f)
        pids = []
        for i in range(len(self.p_tracks['ped_id'])):
            track = self.p_tracks['ped_id'][i]
            if len(track) == 0:
                continue
            try:
                pids.append(track[0][0])
            except:
                import pdb;pdb.set_trace()
                raise ValueError()

        self.samples = self.split_to_samples(self.p_tracks)
        print('-----------Convert samples to ndarray----------')
        for k in tqdm(self.samples.keys()):
            self.samples[k] = np.array(self.samples[k])
        self.num_samples = len(self.samples['obs_image_paths'])
        # get neighbors
        if 'social' in self.modalities:
            self.neighbor_seq_path = os.path.join(self.extra_root,
                f'{self.dataset_name}_fps_{self.fps}_obs_{self.obs_len}_pred_{self.pred_len}_interval_{self.seq_interval}_overlap_{self.overlap_ratio}.pkl')
            self.samples = self.get_neighbor_relation(self.samples,
                                                self.neighbor_seq_path)
        # remove small bbox
        if min_wh is not None:
            self.samples = self.rm_small_bb(self.samples, min_wh)
        # remove occlusion
        if max_occ < 2:
            self.samples = self.rm_occluded(self.samples, max_occ)

        # apply interval
        if self.seq_interval > 0:
            self.downsample_seq()
            print('Applied interval')
            print('cur input len', len(self.samples['obs_img_nm_int'][0]))

        # small set
        assert small_set <= 1, small_set
        if small_set > 0:
            small_set_size = int(self.num_samples * small_set)
            for k in self.samples.keys():
                self.samples[k] = self.samples[k][:small_set_size]
            self.num_samples = small_set_size
        # balance
        if do_balance:
            self.samples = self.balance(balance_label='target', 
                                        all_samples=self.samples)
            print('num samples before balance:', self.num_samples)
            self.num_samples = len(self.samples['obs_image_paths'])
            print('num samples after balance:', self.num_samples)
        
        # add seg maps in samples
        if 'ctx' in self.modalities:
            if self.ctx_format in ('ped_graph', 'ped_graph_seg'):
                ctx_format_dir = 'ori_local'
            else:
                ctx_format_dir = self.ctx_format
            self.ctx_path = os.path.join(self.root_path, 
                                         'context', 
                                         ctx_format_dir, 
                                         str(ctx_size[0])+'w_by_'\
                                            +str(ctx_size[1])+'h')
            if self.ctx_format == 'seg_multi' \
                or self.ctx_format == 'seg_single' \
                    or self.ctx_format == 'local_seg_multi':
                self.ctx_path = os.path.join(self.root_path, 'seg')
                seg_paths = []
                for s in range(self.num_samples):
                    img_path_seq = self.samples['obs_image_paths'][s]
                    seg_path_seq = []
                    for img_path in img_path_seq:
                        seg_path = img_path.replace('images', 'seg')\
                            .replace('.png', '.pkl')
                        seg_path_seq.append(seg_path)
                    seg_paths.append(seg_path_seq)
                self.samples['obs_seg_paths'] = seg_paths

        if self.pred_context:
            self.pred_context_path = \
                os.path.join(self.root_path, 
                             'context', 
                             self.pred_context_mode, 
                             str(ctx_size[0])+'w_by_'+str(ctx_size[1])+'h')
        
        # augmentation
        self.samples = self._add_augment(self.samples)

        print('Total samples: ', self.num_samples)
        # check num samles
        for k in self.samples:
            # print(k)
            assert len(self.samples[k]) == self.num_samples, \
                (k, len(self.samples[k]), self.num_samples)

        # class count
        # import pdb;pdb.set_trace()
        self.n_c = sum(np.squeeze(self.samples['target']))
        self.n_nc = self.num_samples - self.n_c
        self.num_samples_cls = [self.n_nc, self.n_c]
        self.class_weights = {}
        self.class_weights['cross'] = cls_weights(self.num_samples_cls, 'sklearn')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # print('-----------getting item-----------')
        # import pdb; pdb.set_trace()
        obs_bbox_offset = copy.deepcopy(torch.tensor(self.samples['obs_bbox_normed'][idx]).float())  # ltrb
        obs_bbox = copy.deepcopy(torch.tensor(self.samples['obs_bbox'][idx]).float())  # ltrb
        pred_bbox_offset = copy.deepcopy(torch.tensor(self.samples['pred_bbox_normed'][idx]).float())
        pred_bbox = copy.deepcopy(torch.tensor(self.samples['pred_bbox'][idx]).float())
        target = torch.tensor(self.samples['target'][idx][-1])
        obs_ego = \
            torch.tensor(self.samples['obs_ego'][idx]).float().reshape(-1, 1)
        assert len(obs_ego.shape) == 2, obs_ego.shape
        set_id_int = self.samples['obs_set_id_int'][idx][-1] if self.dataset_name == 'PIE' else 0

        # obs_ego = torch.cat([obs_ego, torch.zeros(obs_ego.size())], dim=-1)
        obs_bbox_ori = copy.deepcopy(obs_bbox)
        pred_bbox_ori = copy.deepcopy(pred_bbox)
        # normalize the coordinates
        if '0-1' in self.traj_format:
            obs_bbox_offset = norm_bbox(obs_bbox_offset, self.dataset_name)
            pred_bbox_offset = norm_bbox(pred_bbox_offset, self.dataset_name)
            obs_bbox = norm_bbox(obs_bbox, self.dataset_name)
            pred_bbox = norm_bbox(pred_bbox, self.dataset_name)

        sample = {'dataset_name': torch.tensor(DATASET_TO_ID[self.dataset_name]),
                'set_id_int': torch.tensor(set_id_int),  # obs_len,
                'vid_id_int': torch.tensor(self.samples['obs_vid_id_int'][idx][-1]),  # int
                'ped_id_int': torch.tensor(self.samples['obs_ped_id_int'][idx][-1]),  # int out of (set id, vid id, ped id)
                'img_nm_int': torch.tensor(self.samples['obs_img_nm_int'][idx]),  # obs_len,
                'obs_bboxes':obs_bbox_offset, # obslen, 4
                'obs_bboxes_unnormed': obs_bbox,
                'obs_bboxes_ori': obs_bbox_ori,
                'obs_ego': obs_ego,  # shape: obs len, 1
                'pred_act': target,   # int
                'pred_bboxes': pred_bbox_offset,   # shape: [pred_len, 4]
                'pred_bboxes_unnormed': pred_bbox,
                'pred_bboxes_ori': pred_bbox_ori,
                'atomic_actions': torch.tensor(0),
                'simple_context': torch.tensor(0),
                'complex_context': torch.tensor(0),  # (1,)
                'communicative': torch.tensor(0),
                'transporting': torch.tensor(0),
                'age': torch.tensor(0),
                'hflip_flag': torch.tensor(0),
                'img_ijhw': torch.tensor([-1, -1, -1, -1]),
                'ctx_ijhw': torch.tensor([-1, -1, -1, -1]),
                'sklt_ijhw': torch.tensor([-1, -1, -1, -1]),
                'obs_neighbor_relation': torch.zeros((1, self.obs_len, 5)),
                'obs_neighbor_bbox': torch.zeros((1, self.obs_len, 4)),
                'obs_neighbor_oid': torch.zeros((1,)),
                }

        if 'social' in self.modalities:
            relations, neighbor_bbox, neighbor_oid =\
                  pad_neighbor([copy.deepcopy(np.array(self.samples['obs_neighbor_relations'][idx]).transpose(1,0,2)),
                                copy.deepcopy(np.array(self.samples['obs_neighbor_bbox'][idx]).transpose(1,0,2)),
                                copy.deepcopy(self.samples['obs_neighbor_oid'][idx])
                                ],
                                self.max_n_neighbor)
            if self.social_format == 'rel_loc':
                sample['obs_neighbor_relation'] = torch.tensor(relations).float()  # K T 5
            elif self.social_format == 'ori_traj':
                sample['obs_neighbor_relation'] =\
                      torch.cat([obs_bbox_ori.unsqueeze(0), torch.tensor(neighbor_bbox).float()], 0) # K+1 T 4
            sample['obs_neighbor_bbox'] = torch.tensor(neighbor_bbox).float()
            sample['obs_neighbor_oid'] = torch.tensor(neighbor_oid).float()
        if 'sklt' in self.modalities:
            if 'coord' in self.sklt_format:
                pid = self.samples['obs_pid'][idx][0][0]
                coords = []
                ori_obs_img_paths = self.samples['obs_image_paths'][idx]
                for path in ori_obs_img_paths:
                    img_nm = path.split('/')[-1]
                    coord_nm = img_nm.replace('.png', '.pkl')
                    coord_path = os.path.join(self.sk_coord_path, pid, coord_nm)
                    with open(coord_path, 'rb') as f:
                        coord = pickle.load(f)  # 17, 3
                    coords.append(coord[:, :2])  # 17, 2
                coords = np.stack(coords, axis=0)  # T 17, 2
                pred_coords = []
                ori_pred_img_paths = self.samples['pred_image_paths'][idx]
                for path in ori_pred_img_paths:
                    img_nm = path.split('/')[-1]
                    coord_nm = img_nm.replace('.png', '.pkl')
                    coord_path = os.path.join(self.sk_coord_path, pid, coord_nm)
                    with open(coord_path, 'rb') as f:
                        coord = pickle.load(f)  # 17, 3
                    pred_coords.append(coord[:, :2])  # 17, 2
                pred_coords = np.stack(pred_coords, axis=0)  # T 17, 2
                # (T, N, C) -> (C, T, N)
                obs_skeletons = torch.from_numpy(coords).float().permute(2, 0, 1)  # 2, T, 17
                pred_skeletons = torch.from_numpy(pred_coords).float().permute(2, 0, 1)
                # yx -> xy
                obs_skeletons = obs_skeletons.flip(0)
                pred_skeletons = pred_skeletons.flip(0)
                # add offset
                obs_skeletons = sklt_local_to_global(obs_skeletons.float(), obs_bbox_ori.float())
                pred_skeletons = sklt_local_to_global(pred_skeletons.float(), pred_bbox_ori.float())
                if '0-1' in self.sklt_format:
                    obs_skeletons = norm_sklt(obs_skeletons, self.dataset_name)
                    pred_skeletons = norm_sklt(pred_skeletons, self.dataset_name)
            elif self.sklt_format == 'heatmap':
                pid = self.samples['obs_pid'][idx][0][0]
                heatmaps = []
                ori_obs_img_paths = self.samples['obs_image_paths'][idx]
                for path in ori_obs_img_paths:
                    img_nm = path.split('/')[-1]
                    heatmap_nm = img_nm.replace('.png', '.pkl')
                    heatmap_path = os.path.join(self.sk_heatmap_path, pid, heatmap_nm)
                    with open(heatmap_path, 'rb') as f:
                        heatmap = pickle.load(f)
                    heatmaps.append(heatmap)
                heatmaps = np.stack(heatmaps, axis=0)  # T C H W
                # T C H W -> C T H W
                obs_skeletons = torch.from_numpy(heatmaps).float().permute(1, 0, 2, 3)  # shape: (17, seq_len, 96, 72)
                pred_heatmaps = []
                ori_pred_img_paths = self.samples['pred_image_paths'][idx]
                for path in ori_pred_img_paths:
                    img_nm = path.split('/')[-1]
                    heatmap_nm = img_nm.replace('.png', '.pkl')
                    heatmap_path = os.path.join(self.sk_heatmap_path, pid, heatmap_nm)
                    with open(heatmap_path, 'rb') as f:
                        heatmap = pickle.load(f)
                    pred_heatmaps.append(heatmap)
                pred_heatmaps = np.stack(pred_heatmaps, axis=0)  # T C H W
                # T C H W -> C T H W
                pred_skeletons = torch.from_numpy(pred_heatmaps).float().permute(1, 0, 2, 3)
            elif self.sklt_format == 'pseudo_heatmap':
                pid = self.samples['obs_pid'][idx][0][0]
                heatmaps = []
                ori_obs_img_paths = self.samples['obs_image_paths'][idx]
                for path in ori_obs_img_paths:
                    img_nm = path.split('/')[-1]
                    heatmap_nm = img_nm.replace('.png', '.pkl')
                    heatmap_path = os.path.join(self.sk_p_heatmap_path, pid, heatmap_nm)
                    with open(heatmap_path, 'rb') as f:
                        heatmap = pickle.load(f)
                    heatmaps.append(heatmap)
                heatmaps = np.stack(heatmaps, axis=0)  # T C H W
                # T C H W -> C T H W
                obs_skeletons = torch.from_numpy(heatmaps).float().permute(1, 0, 2, 3)  # shape: (17, seq_len, 48, 48)
                pred_heatmaps = []
                ori_pred_img_paths = self.samples['pred_image_paths'][idx]
                for path in ori_pred_img_paths:
                    img_nm = path.split('/')[-1]
                    heatmap_nm = img_nm.replace('.png', '.pkl')
                    heatmap_path = os.path.join(self.sk_p_heatmap_path, pid, heatmap_nm)
                    with open(heatmap_path, 'rb') as f:
                        heatmap = pickle.load(f)
                    pred_heatmaps.append(heatmap)
                pred_heatmaps = np.stack(pred_heatmaps, axis=0)  # T C H W
                # T C H W -> C T H W
                pred_skeletons = torch.from_numpy(pred_heatmaps).float().permute(1, 0, 2, 3)
            elif self.sklt_format == 'img+heatmap':
                raise NotImplementedError()
                if not self.crop_from_ori:
                    pid = self.samples['obs_pid'][idx][0][0]
                    imgs = []
                    ori_obs_img_paths = self.samples['obs_image_paths'][idx]
                    for path in ori_obs_img_paths:
                        img_nm = path.split('/')[-1]
                        img_path = os.path.join(self.cropped_path, pid, img_nm)
                        imgs.append(cv2.imread(img_path))
                else:
                    obs_img_paths = self.samples['obs_image_paths'][idx]
                    obs_bbox_offset = self.samples['obs_bbox'][idx]
                    imgs = []
                    for path, bbox in zip(obs_img_paths, obs_bbox_offset):
                        imgs.append(self.load_cropped_image(path, bbox))
                # (T, H, W, C) -> (C, T, H, W)
                imgs = np.stack(imgs, axis=0)
                ped_imgs = torch.from_numpy(imgs).float().permute(3, 0, 1, 2)
                # normalize img
                if self.img_norm_mode != 'ori':
                    ped_imgs = norm_imgs(ped_imgs, self.img_mean, self.img_std)
                # BGR -> RGB
                if self.target_color_order == 'RGB':
                    # ped_imgs = torch.from_numpy(np.ascontiguousarray(ped_imgs.numpy()[::-1, :, :, :]))  # 3 T H W
                    ped_imgs = torch.flip(ped_imgs, dims=[0])
                # load heatmaps
                pid = self.samples['obs_pid'][idx][0][0]
                heatmaps = []
                ori_obs_img_paths = self.samples['obs_image_paths'][idx]
                for path in ori_obs_img_paths:
                    img_nm = path.split('/')[-1]
                    heatmap_nm = img_nm.replace('.png', '.pkl')
                    heatmap_path = os.path.join(self.sk_heatmap_path, pid, heatmap_nm)
                    with open(heatmap_path, 'rb') as f:
                        heatmap = pickle.load(f)
                    heatmaps.append(heatmap)
                heatmaps = np.stack(heatmaps, axis=0)  # T C H W
                # T C H W -> C T H W
                heatmaps = torch.from_numpy(heatmaps).float().permute(1, 0, 2, 3)  # shape: (17, seq_len, 96, 72)
                heatmaps = torch.sum(heatmaps, dim=0, keepdim=True)  # shape: (1, seq_len, 96, 72)
                # normalize heatmaps
                heatmaps = heatmaps - torch.amin(heatmaps, dim=(2, 3), keepdim=True)
                heatmaps = heatmaps / torch.amax(heatmaps, dim=(2, 3), keepdim=True)
                # interpolate
                heatmaps = F.interpolate(heatmaps, size=(ped_imgs.size(2),
                                                         ped_imgs.size(3)), mode='bilinear', align_corners=True)  # 1 T H W
                obs_skeletons = heatmaps * ped_imgs
            else:
                raise NotImplementedError(self.sklt_format)
            sample['obs_skeletons'] = obs_skeletons
            sample['pred_skeletons'] = pred_skeletons
        
        if 'img' in self.modalities:
            # print('-----------getting img-----------')
            if self.crop_from_ori:
                obs_img_paths = self.samples['obs_image_paths'][idx]
                imgs = []
                for path, bbox in zip(obs_img_paths, obs_bbox_ori):
                    imgs.append(self.load_cropped_image(path, bbox))
            else:
                pid = self.samples['obs_pid'][idx][0][0]
                imgs = []
                ori_obs_img_paths = self.samples['obs_image_paths'][idx]
                for path in ori_obs_img_paths:
                    img_nm = path.split('/')[-1]
                    img_path = os.path.join(self.cropped_path, pid, img_nm)
                    imgs.append(cv2.imread(img_path))
            # temporal stack
            imgs = np.stack(imgs, axis=0)
            # (T, H, W, C) -> (C, T, H, W)
            try:
                ped_imgs = torch.from_numpy(imgs).float().permute(3, 0, 1, 2)
            except:
                import pdb;pdb.set_trace()

            # normalize img
            if self.img_norm_mode != 'ori':
                ped_imgs = norm_imgs(ped_imgs, self.img_mean, self.img_std)
            # BGR -> RGB
            if self.target_color_order == 'RGB':
                # ped_imgs = torch.from_numpy(np.ascontiguousarray(ped_imgs.numpy()[::-1, :, :, :]))
                ped_imgs = torch.flip(ped_imgs, dims=[0])
            sample['ped_imgs'] = ped_imgs  # shape [3, obs_len, H, W]

        if 'ctx' in self.modalities:
            # print('-----------getting ctx-----------')
            if self.ctx_format in ('mask_ped', 'local', 'ori_local', 'ori', 
                 'ped_graph', 'ped_graph_seg'):
                pid = self.samples['obs_pid'][idx][0][0]
                setid, vidid, oid = pid.split('_')
                ctx_imgs = []
                ori_obs_img_paths = self.samples['obs_image_paths'][idx]
                for path in ori_obs_img_paths:
                    img_nm = path.split('/')[-1]
                    img_path = os.path.join(self.ctx_path, pid, img_nm)
                    ctx_imgs.append(cv2.imread(img_path))
                ctx_imgs = np.stack(ctx_imgs, axis=0)
                ctx_imgs = torch.from_numpy(ctx_imgs).float().permute(3, 0, 1, 2)  # 3 t h w
                # normalize img
                if self.img_norm_mode != 'ori':
                    ctx_imgs = norm_imgs(ctx_imgs, 
                                         self.img_mean, self.img_std)
                # BGR -> RGB
                if self.target_color_order == 'RGB':
                   ctx_imgs = torch.flip(ctx_imgs, dims=[0])
                # load cropped seg
                if self.ctx_format in ('ped_graph', 'ped_graph_seg'):
                    if self.dataset_name == 'PIE':
                        vid_dir = '/'.join([setid, vidid])
                    else:
                        vid_dir = vidid
                    img_nm = ori_obs_img_paths[-1].split('/')[-1]
                    all_c_seg = []
                    for c in self.seg_cls:
                        seg_path = os.path.join(self.root_path,
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
                sample['obs_context'] = ctx_imgs  # shape [3, obs_len, H, W]
            elif self.ctx_format in \
                ('seg_ori_local', 'seg_local'):
                # load imgs
                ori_obs_img_paths = self.samples['obs_image_paths'][idx]
                ctx_imgs = []
                for path in ori_obs_img_paths:
                    ctx_imgs.append(cv2.imread(path))
                assert ctx_imgs[0].shape[0] == 1080 and ctx_imgs[0].shape[1] == 1920, ctx_imgs[0].shape
                # except AttributeError:
                #     import pdb
                #     pdb.set_trace()
                #     raise ValueError
                ctx_imgs = np.stack(ctx_imgs, axis=0)  # THWC
                # permute
                ctx_imgs = torch.from_numpy(ctx_imgs).float().permute(3, 0, 1, 2)  # CTHW
                # norm imgs
                if self.img_norm_mode != 'ori':
                    ctx_imgs = norm_imgs(ctx_imgs, 
                                         self.img_mean, 
                                         self.img_std)
                # BGR -> RGB
                if self.target_color_order == 'RGB':
                    ctx_imgs = torch.flip(ctx_imgs, dims=[0])  # CTHW
                # load seg maps
                ctx_segs = {c:[] for c in self.seg_cls}
                if self.dataset_name == 'PIE':
                    for pth in ori_obs_img_paths:
                        s_nm, v_nm, i_nm = pth.split('/')[-3:]
                        s_id, v_id, f_nm = str(int(s_nm.replace('set0', ''))), \
                                            str(int(v_nm.replace('video_', ''))), \
                                                i_nm.replace('png', 'pkl')
                        for c in self.seg_cls:
                            seg_path = os.path.join(self.root_path, 
                                                    c, 
                                                    s_id, 
                                                    v_id, 
                                                    f_nm)
                            with open(seg_path, 'rb') as f:
                                seg = pickle.load(f)
                            ctx_segs[c].append(torch.from_numpy(seg))
                elif self.dataset_name == 'JAAD':
                    for pth in ori_obs_img_paths:
                        v_nm, i_nm = pth.split('/')[-2:]
                        v_id, f_nm = str(int(v_nm.replace('video_', ''))), i_nm.replace('png', 'pkl')
                        for c in self.seg_cls:
                            seg_path = os.path.join(self.sam_seg_root, c, v_id, f_nm)
                            with open(seg_path, 'rb') as f:
                                seg = pickle.load(seg_path)
                            ctx_segs[c].append(torch.from_numpy(seg))
                for c in self.seg_cls:
                    ctx_segs[c] = torch.stack(ctx_segs[c], dim=0)  # THW
                # crop
                crop_imgs = []
                crop_segs = {c:[] for c in self.seg_cls}
                for i in range(ctx_imgs.size(1)):  # T
                    crop_img = crop_local_ctx(ctx_imgs[:, i], obs_bbox_ori[i], self.ctx_size)  # 3 h w
                    crop_imgs.append(crop_img)
                    for c in self.seg_cls:
                        crop_seg = crop_local_ctx(torch.unsqueeze(ctx_segs[c][i], dim=0), obs_bbox_ori[i], self.ctx_size)  # 1 h w
                        crop_segs[c].append(crop_seg)
                crop_imgs = torch.stack(crop_imgs, dim=1)  # 3Thw
                all_seg = []
                for c in self.seg_cls:
                    all_seg.append(torch.stack(crop_segs[c], dim=1))  # 1Thw
                all_seg = torch.stack(all_seg, dim=4)  # 1Thw n_cls
                if self.ctx_format == 'ped_graph':
                    all_seg = torch.argmax(all_seg[0, -1], dim=-1, keepdim=True).permute(2, 0, 1)  # 1 h w
                    sample['obs_context'] = torch.concat([crop_imgs[:, -1], all_seg], dim=0)
                else:
                    sample['obs_context'] = all_seg * torch.unsqueeze(crop_imgs, dim=-1)  # 3Thw n_cls
            else:
                raise ValueError(self.ctx_format)
        
        if self.pred_context:
            if self.pred_context_mode == 'ori':
                pid = self.samples['obs_pid'][idx][0][0]
                ctx_imgs = []
                ori_obs_img_paths = self.samples['pred_image_paths'][idx]
                for path in ori_obs_img_paths:
                    img_nm = path.split('/')[-1]
                    img_path = os.path.join(self.ctx_path, pid, img_nm)
                    ctx_imgs.append(cv2.imread(img_path))
                ctx_imgs = np.stack(ctx_imgs, axis=0)
                ctx_imgs = torch.from_numpy(ctx_imgs).float().permute(3, 0, 1, 2)
                # normalize img
                if self.img_norm_mode != 'ori':
                    ctx_imgs = norm_imgs(ctx_imgs, self.img_mean, self.img_std)
                # BGR -> RGB
                if self.target_color_order == 'RGB':
                    ctx_imgs = torch.from_numpy(np.ascontiguousarray(ctx_imgs.numpy()[::-1, :, :, :]))
                sample['pred_context'] = ctx_imgs  # shape [3, obs_len, H, W]

        # augmentation
        if self.augment_mode != 'none':
            if self.transforms['random']:
                sample = self._ranndom_augment(sample)
            elif self.transforms['balance']:
                sample['hflip_flag'] = torch.tensor(self.samples['hflip_flag'][idx])
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
                    RandomResizedCrop(size=self.bbox_size, # (h, w)
                                        scale=(0.75, 1), 
                                        ratio=(self.bbox_size[1]/self.bbox_size[0], 
                                                self.bbox_size[1]/self.bbox_size[0]))  # w / h
            if 'ctx' in self.modalities:
                self.transforms['resized_crop']['ctx'] = RandomResizedCrop(size=self.ctx_size, # (h, w)
                                                                        scale=(0.75, 1), 
                                                                        ratio=(self.ctx_size[1]/self.ctx_size[0], self.ctx_size[1]/self.ctx_size[0]))  # w / h
            if 'sklt' in self.modalities and self.sklt_format == 'pseudo_heatmap':
                self.transforms['resized_crop']['sklt'] = RandomResizedCrop(size=(48, 48), # (h, w)
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

                # init extra samples
                h_flip_samples = {}
                for k in data:
                    h_flip_samples[k] = []

                # duplicate samples
                for i in range(len(data['obs_image_paths'])):
                    if data['target'][i][0] == 1:
                        for k in data:
                            h_flip_samples[k].append(copy.deepcopy(data[k][i]))
                h_flip_samples['hflip_flag'] = [True] * len(h_flip_samples['obs_image_paths'])
                data['hflip_flag'] = np.array([False] * len(data['obs_image_paths']))
                # concat
                for k in data:
                    data[k] = np.concatenate([data[k], np.array(h_flip_samples[k])], axis=0)

            self.num_samples = len(data['obs_image_paths'])
            print(f'Num samples after flip: {self.num_samples}')
        return data

    def _augment(self, sample):
        # flip
        if sample['hflip_flag']:
            if 'img' in self.modalities:
                sample['ped_imgs'] = TVF.hflip(sample['ped_imgs'])
            if 'ctx' in self.modalities:
                sample['obs_context'] = TVF.hflip(sample['obs_context'])
            if 'sklt' in self.modalities and ('heatmap' in self.sk_mode):
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
            sample['sk_ijhw'] = torch.tensor(ijhw)
        return sample

    def _ranndom_augment(self, sample):
        # flip
        if self.transforms['hflip'] is not None:
            sample['hflip_flag'] = torch.tensor(self.transforms['hflip'].flag)
            if 'img' in self.modalities:
                sample['ped_imgs'] = self.transforms['hflip'](sample['ped_imgs'])
            if 'ctx' in self.modalities:
                sample['obs_context'] = self.transforms['hflip'](sample['obs_context'])
            if 'sklt' in self.modalities and ('heatmap' in self.sklt_format):
                sample['obs_skeletons'] = self.transforms['hflip'](sample['obs_skeletons'])
            if 'traj' in self.modalities and self.transforms['hflip'].flag:
                sample['obs_bboxes_unnormed'][:, 0], sample['obs_bboxes_unnormed'][:, 2] = \
                    1920 - sample['obs_bboxes_unnormed'][:, 2], 1920 - sample['obs_bboxes_unnormed'][:, 0]
                if '0-1' in self.traj_format:
                    sample['obs_bboxes'][:, 0], sample['obs_bboxes'][:, 2] =\
                         1 - sample['obs_bboxes'][:, 2], 1 - sample['obs_bboxes'][:, 0]
                else:
                    sample['obs_bboxes'][:, 0], sample['obs_bboxes'][:, 2] =\
                         1920 - sample['obs_bboxes'][:, 2], 1920 - sample['obs_bboxes'][:, 0]
            self.transforms['hflip'].randomize_parameters()
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
            sample['sk_ijhw'] = torch.tensor(ijhw)
        return sample
    
    def _convert_speed_unit(self, ego_speed_tracks):
        '''
        Convert speed unit from km/h to m/s
        '''
        new_tracks = []
        for track in ego_speed_tracks:
            new_tracks.append([])
            for speed in track:
                speed = speed[0]
                new_tracks[-1].append(speed / 3.6)
        return new_tracks
    
    def _get_accel(self, p_tracks):
        '''
        Add acceleration to the keys of tracks.
        The lengths of all tracks will be substracted by 1.
        '''
        new_tracks = {'ego_accel':[]}
        for speed_track in p_tracks['obd_speed']:
            new_tracks['ego_accel'].append([])
            for i in range(len(speed_track) - 1):
                accel = (speed_track[i+1] - speed_track[i]) * 30  # 30 fps
                if self.speed_unit == 'km/h':
                    accel = accel / 3.6
                new_tracks['ego_accel'][-1].append(accel)

        # rest keys
        # pop the last frame for every track
        for k in p_tracks:
            new_tracks[k] = []
            for i in range(len(p_tracks[k])):
                new_tracks[k].append(p_tracks[k][i][:-1])
                assert len(new_tracks[k][i]) == len(new_tracks['ego_accel'][i]), \
                    (len(new_tracks[k][i]), len(new_tracks['ego_accel'][i]))
        return new_tracks

    def _get_v_tracks(self):
        pass

    def get_neighbor_relation(self,
                      samples,
                      save_path,
                      padding_val=0):
        if os.path.exists(save_path) and False:
            with open(save_path, 'rb') as f:
                neighbor_seq = pickle.load(f)
        else:
            n_sample = len(samples['obs_bbox'])
            relations = []
            neighbor_bbox = []
            neighbor_oid = []
            neighbor_cls = []
            print('Getting neighbor sequences')
            for i in tqdm(range(n_sample)):
                target_cid = str(samples['obs_vid_id_int'][i][0])  # str
                target_oid = str(samples['obs_ped_id_int'][i][-1])  # str (sid vid oid)
                if self.dataset_name == 'PIE':
                    target_sid = str(samples['obs_set_id_int'][i][0])  # str
                    vid_dict = self.imgnm_to_objid[target_sid][target_cid]
                else:
                    vid_dict = self.imgnm_to_objid[target_cid]
                target_bbox_seq = np.array(samples['obs_bbox'][i])  # T, 4
                obslen = len(target_bbox_seq)
                bbox_seq_dict = {'ped':{},
                                'veh':{}}
                for j in range(obslen):
                    imgnm = samples['obs_image_paths'][i][j].split('/')[-1]  # str
                    cur_ped_ids = set(vid_dict[imgnm]['ped'].keys())
                    try:
                        cur_ped_ids.remove(target_oid)
                    except:
                        # print(cur_ped_ids, target_oid, target_sid, target_cid, imgnm, self.dataset_name)
                        import pdb; pdb.set_trace()
                        raise ValueError()
                    cur_veh_ids = set(vid_dict[imgnm]['veh'].keys())
                    # ped neighbor for cur sample
                    # existing neighbor
                    for oid in set(bbox_seq_dict['ped'].keys())&cur_ped_ids:
                        bbox = np.array(vid_dict[imgnm]['ped'][oid]['bbox'])
                        bbox_seq_dict['ped'][oid].append(bbox)
                    # first appearing neighbor
                    for oid in cur_ped_ids-set(bbox_seq_dict['ped'].keys()):
                        bbox = np.array(vid_dict[imgnm]['ped'][oid]['bbox'])
                        bbox_seq_dict['ped'][oid] = [np.ones([4])*padding_val]*j + [bbox]  # T, 4
                    # disappeared neighbor
                    for oid in set(bbox_seq_dict['ped'].keys())-cur_ped_ids:
                        bbox_seq_dict['ped'][oid].append(np.ones([4])*padding_val)
                    
                    # veh neighbor for cur sample
                    # existing neighbor
                    for oid in set(bbox_seq_dict['veh'].keys())&cur_veh_ids:
                        bbox = np.array(vid_dict[imgnm]['veh'][oid]['bbox'])
                        bbox_seq_dict['veh'][oid].append(bbox)
                    # first appearing neighbor
                    for oid in cur_veh_ids-set(bbox_seq_dict['veh'].keys()):
                        bbox = np.array(vid_dict[imgnm]['veh'][oid]['bbox'])
                        bbox_seq_dict['veh'][oid] = [np.ones([4])*padding_val]*j + [bbox]  # T, 4
                    # disappeared neighbor
                    for oid in set(bbox_seq_dict['veh'].keys())-cur_veh_ids:
                        bbox_seq_dict['veh'][oid].append(np.ones([4])*padding_val)
                cur_neighbor_bbox = []
                cur_neighbor_oid = []
                cur_neighbor_cls = []
                for oid in bbox_seq_dict['ped']:
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
            neighbor_seq['obs_neighbor_relations'] = relations
            neighbor_seq['obs_neighbor_bbox'] = neighbor_bbox
            neighbor_seq['obs_neighbor_oid'] = neighbor_oid
            neighbor_seq['obs_neighbor_cls'] = neighbor_cls
            with open(save_path, 'wb') as f:
                pickle.dump(neighbor_seq, f)
        # add neighbor info to samples
        for k in neighbor_seq:
            samples[k] = neighbor_seq[k]
        return samples
    
    def get_imgnm_to_objid(self, p_tracks, v_tracks, save_path):
        # cid_to_imgnm_to_oid_to_info: cid -> img name -> obj type (ped/veh) -> obj id -> bbox/ego motion
        imgnm_to_oid_to_info = {}
        
        if self.dataset_name == 'PIE':
            # pedestrian tracks
            print(f'Saving imgnm to objid to obj info of pedestrians in \
                {self.dataset_name}')
            tracks = p_tracks
            n_tracks = len(tracks['image'])
            # track loop
            for i in range(n_tracks):
                img_paths = tracks['image'][i]
                ped_id = tracks['ped_id'][i][0][0]
                set_id, vid_id, oid = ped_id.split('_')
                # initialize set and video dict
                if set_id not in imgnm_to_oid_to_info:
                    imgnm_to_oid_to_info[set_id] = {}
                if vid_id not in imgnm_to_oid_to_info[set_id]:
                    imgnm_to_oid_to_info[set_id][vid_id] = {}
                # image loop
                for j in range(len(img_paths)):
                    img_path = img_paths[j]
                    img_nm = img_path.split('/')[-1]
                    # initialize img dict
                    if img_nm not in imgnm_to_oid_to_info[set_id][vid_id]:
                        imgnm_to_oid_to_info[set_id][vid_id][img_nm] = {}
                        imgnm_to_oid_to_info[set_id][vid_id][img_nm]['ped'] = {}
                        imgnm_to_oid_to_info[set_id][vid_id][img_nm]['veh'] = {}
                    # initialize obj dict
                    imgnm_to_oid_to_info[set_id][vid_id]\
                        [img_nm]['ped'][oid] = {}
                    # add obj info
                    bbox = tracks['bbox'][i][j]  # ltrb float
                    bbox = list(map(int, bbox))  # int
                    imgnm_to_oid_to_info[set_id][vid_id][img_nm]['ped']\
                        [oid]['bbox'] = bbox
            # vehicle tracks
            print(f'Saving imgnm to objid to obj info of vehicles in \
                {self.dataset_name}')
            tracks = v_tracks
            for set_id in v_tracks:
                if set_id not in imgnm_to_oid_to_info:
                    imgnm_to_oid_to_info[set_id] = {}
                for vid_id in v_tracks[set_id]:
                    if vid_id not in imgnm_to_oid_to_info[set_id]:
                        imgnm_to_oid_to_info[set_id][vid_id] = {}
                    for oid in v_tracks[set_id][vid_id]:
                        # one vehcle might have multiple tracks
                        for i in range(len(v_tracks[set_id]\
                                           [vid_id][oid]['img_nm'])):
                            img_nms = v_tracks[set_id][vid_id][oid]['img_nm'][i]
                            bboxes = v_tracks[set_id][vid_id][oid]['bbox'][i]
                            assert len(img_nms) == len(bboxes), \
                                (len(img_nms), len(bboxes))
                            for j in range(len(img_nms)):
                                img_nm = img_nms[j]
                                bbox = bboxes[j]
                                bbox = list(map(int, bbox))  # int
                                if img_nm not in imgnm_to_oid_to_info[set_id]\
                                    [vid_id]:
                                    imgnm_to_oid_to_info[set_id]\
                                        [vid_id][img_nm] = {}
                                    imgnm_to_oid_to_info[set_id]\
                                        [vid_id][img_nm]['ped'] = {}
                                    imgnm_to_oid_to_info[set_id]\
                                        [vid_id][img_nm]['veh'] = {}
                                # initialize obj dict
                                imgnm_to_oid_to_info[set_id]\
                                    [vid_id][img_nm]['veh'][oid] = {}
                                # add obj info
                                imgnm_to_oid_to_info[set_id][vid_id]\
                                    [img_nm]['veh'][oid]['bbox'] = bbox
        elif self.dataset_name == 'JAAD':
            raise ValueError('JAAD is discarded')
        # save the file
        with open(save_path, 'wb') as f:
            pickle.dump(imgnm_to_oid_to_info, f)
        
        return imgnm_to_oid_to_info
        
    def split_to_samples(self, data):
        seq_length = self._obs_len + self._pred_len
        overlap_stride = self._obs_len if self.overlap_ratio == 0 \
            else int((1 - self.overlap_ratio) * self._obs_len)
        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        #  Check data types
        data.pop('image_dimension', None)
        samples = {}
        for dt in data.keys():
            try:
                samples[dt] = data[dt]
            except KeyError:
                raise ('Wrong data type is selected %s' % dt)

        samples['image'] = data['image']
        samples['ped_id'] = data['ped_id']

        #  Sample tracks into samples
        print('---------------Split tracks to samples---------------')
        print(samples.keys())
        for k in tqdm(samples.keys()):
            # import pdb;pdb.set_trace()
            tracks = []
            for track in samples[k]:
                # skip too short 
                if len(track) < seq_length:
                    continue
                if self.tte is not None:
                    if len(track) < self.tte[1]:
                        continue
                    start_idx = len(track) - self.tte[1]
                    end_idx = len(track) - self.tte[0] - seq_length
                    tracks.extend([track[i:i + seq_length] for i in range(start_idx, end_idx, overlap_stride)])
                else:
                    tracks.extend([track[i:i + seq_length] for i in range(0, len(track) - seq_length, overlap_stride)])
            samples[k] = tracks

        #  Normalize tracks by subtracting bbox/center at first time step from the rest
        print('---------------Normalize traj---------------')
        normed_samples = copy.deepcopy(samples)
        
        if self.offset_traj:
            for i in range(len(normed_samples['bbox'])):
                normed_samples['bbox'][i] = np.subtract(normed_samples['bbox'][i][:],
                                                        normed_samples['bbox'][i][0]).tolist()
            for i in range(len(normed_samples['center'])):
                normed_samples['center'][i] = np.subtract(normed_samples['center'][i][:],
                                                            normed_samples['center'][i][0]).tolist()
        if self.traj_format == 'xywh':
            for i in range(len(normed_samples['bbox'])):
                normed_samples['bbox'][i] = ltrb2xywh(normed_samples['bbox'][i])

        obs_slices = {}
        pred_slices = {}
        obs_bbox_normed = []
        pred_bbox_normed = []
        obs_center_normed = []
        pred_center_normed = []

        #  Split obs and pred
        print('---------------Split obs and pred---------------')
        for k in samples.keys():
            obs_slices[k] = []
            pred_slices[k] = []
            obs_slices[k].extend([d[0:self._obs_len] for d in samples[k]])
            pred_slices[k].extend([d[self._obs_len:] for d in samples[k]])
            if k == 'bbox':
                obs_bbox_normed.extend([d[0:self._obs_len] for d in normed_samples[k]])
                pred_bbox_normed.extend([d[self._obs_len:] for d in normed_samples[k]])

        if self.seq_type == 'crossing':
            obs_slices['activities'] = np.array(obs_slices['activities'])[:, -1]
            pred_slices['activities'] = np.array(pred_slices['activities'])[:, -1]
        else:
            obs_slices['intention_binary'] = np.array(obs_slices['intention_binary'])[:, -1]
            pred_slices['intention_binary'] = np.array(pred_slices['intention_binary'])[:, -1]

        # record ped id, img id
        obs_img_nm_int = []
        ped_id_int = []
        vid_id_int = []
        set_id_int = []
        for i in range(len(obs_slices['image'])):
            pid = obs_slices['ped_id'][i][0][0]
            str_list = pid.split('_')
            int_pid = []
            for s in str_list:
                if 'b' in s:
                    int_pid.append(-int(s.replace('b', '')))
                else:
                    int_pid.append(int(s))
            ped_id_int.append(int_pid)
            int_img_nm_seq = []
            int_vid_nm_seq = []
            int_set_nm_seq = []
            for path in obs_slices['image'][i]:
                nm = path.split('/')[-1].split('.')[0]
                vid_nm = path.split('/')[-2].replace('video_', '')
                int_img_nm_seq.append(int(nm))
                int_vid_nm_seq.append(int(vid_nm))
                if self.dataset_name == 'PIE':
                    set_nm = path.split('/')[-3].replace('set', '')
                    int_set_nm_seq.append(int(set_nm))
                else:
                    int_set_nm_seq.append(0)
            obs_img_nm_int.append(int_img_nm_seq)
            vid_id_int.append(int_vid_nm_seq)
            set_id_int.append(int_set_nm_seq)
        obs_img_nm_int = np.array(obs_img_nm_int)
        ped_id_int = np.array(ped_id_int)
        vid_id_int = np.array(vid_id_int)

        if self.dataset_name == 'PIE':
            set_id_int = np.array(set_id_int)
            all_samples = {'obs_img_nm_int': obs_img_nm_int,
                    'obs_ped_id_int': ped_id_int,
                    'obs_vid_id_int': vid_id_int,
                    'obs_set_id_int': set_id_int,
                    'obs_image_paths': obs_slices['image'],
                    'obs_bbox_normed': obs_bbox_normed,
                    'obs_bbox': obs_slices['bbox'],
                    # 'obs_intent': obs_slices['intention_binary'],
                    'obs_pid': obs_slices['ped_id'],
                    'obs_occ': obs_slices['occlusion'],
                    'obs_ego': obs_slices[self.ego_motion_key],
                    'pred_image_paths': pred_slices['image'],
                    'pred_bbox_normed': pred_bbox_normed,
                    'pred_bbox': pred_slices['bbox'],
                    'pred_ego': pred_slices[self.ego_motion_key],
                    # 'pred_intent': pred_slices['intention_binary'],
                    'pred_pid': pred_slices['ped_id'],
                    'pred_occ': pred_slices['occlusion']}
        elif self.dataset_name == 'JAAD':
            all_samples = {'obs_img_nm_int': obs_img_nm_int,
                'obs_ped_id_int': ped_id_int,
                'obs_vid_id_int': vid_id_int,
                'obs_image_paths': obs_slices['image'],
                'obs_bbox_normed': obs_bbox_normed,
                'obs_bbox': obs_slices['bbox'],
                # 'obs_intent': obs_slices['intention_binary'],
                'obs_pid': obs_slices['ped_id'],
                'obs_occ': obs_slices['occlusion'],
                'obs_ego': obs_slices['vehicle_act'],
                'pred_image_paths': pred_slices['image'],
                'pred_bbox_normed': pred_bbox_normed,
                'pred_bbox': pred_slices['bbox'],
                'pred_ego': pred_slices['vehicle_act'],
                # 'pred_intent': pred_slices['intention_binary'],
                'pred_pid': pred_slices['ped_id'],
                'pred_occ': pred_slices['occlusion']}

        if self.seq_type == 'crossing':
            all_samples['target'] = pred_slices['activities']
            if self.recog_act:
                all_samples['target'] = obs_slices['activities']
        else:
            all_samples['target'] = obs_slices['intention_binary']
        # pdb.set_trace()
        return all_samples

    def downsample_seq(self):
        new_samples = {}
        for k in self.samples:
            if 'obs' in k and len(self.samples[k][0]) == self._obs_len \
                and (k not in ('neighbor_oid', 'neighbor_cls')):
                new_samples[k] = []
                for s in range(len(self.samples[k])):
                    ori_seq = self.samples[k][s]
                    new_seq = []
                    for i in range(0, self._obs_len, self.seq_interval+1):
                        new_seq.append(ori_seq[i])
                    new_samples[k].append(np.array(new_seq))
                    assert len(new_samples[k][s]) == self.obs_len, (k, len(new_samples[k]), self.obs_len)
                if 'neighbor' not in k:
                    new_samples[k] = np.array(new_samples[k])

            if 'pred' in k and len(self.samples[k][0]) == self._pred_len:
                new_samples[k] = []
                for s in range(len(self.samples[k])):
                    ori_seq = self.samples[k][s]
                    new_seq = []
                    for i in range(0, self._pred_len, self.seq_interval+1):
                        new_seq.append(ori_seq[i])
                    new_samples[k].append(np.array(new_seq))
                    assert len(new_samples[k][s]) == self.pred_len, (k, len(new_samples[k]), self.pred_len)
                if 'neighbor' not in k:
                    new_samples[k] = np.array(new_samples[k])
        for k in new_samples:
            # print('new k', k)
            # print(new_samples[k].shape)
            self.samples[k] = new_samples[k]

    def load_cropped_image(self, 
                           img_path, 
                           bbox, 
                           target_size=(224, 224), 
                           resize_mode='resize'):
        '''
        :param img_path:
        :param bbox: l, t, r, d
        :param target_size: W, H
        :param resize_mode:
        :return:
            resized: np array (H, W, 3)
        '''
        t1 = time.time()
        img = cv2.imread(img_path)
        t2 = time.time()
        print('load time', t2-t1)
        l, t, r, b = map(int, bbox)
        cropped = img[t:b, l:r]
        h, w = cropped.shape[0], cropped.shape[1]
        if resize_mode == 'resize':
            resized = cv2.resize(cropped, target_size)
        else:
            if float(target_size[0]) / target_size[1] < float(w) / h:
                ratio = float(target_size[0]) / w
            else:
                ratio = float(target_size[1]) / h
            new_size = (int(w*ratio), int(h*ratio))
            cropped = cv2.resize(cropped, new_size)
            w_pad = target_size[0] - new_size[0]
            h_pad = target_size[1] - new_size[1]
            resized = cv2.copyMakeBorder(cropped,
                                         0,h_pad,0,w_pad,
                                         cv2.BORDER_CONSTANT,value=(0, 0, 0))  # t, b, l, r
        t3 = time.time()
        print('resize time', t3-t2)
        return resized  # H, W, C

    def load_skeletons(self, pids, skeleton_mode='coord'):
        skeleton_tracks = []
        print('------------Loading skeleton data-----------')
        if skeleton_mode == 'coord':
            tbar = tqdm(pids)
            for pid in tbar:
                cur_track_path = os.path.join(self.sk_vis_path, pid, 'skeletons.pkl')
                with open(cur_track_path, 'rb') as f:
                    track = pickle.load(f)  # shape: (track_len, n_j, 3)
                skeleton_tracks.append(track[:, :, :2])
            tbar.close()
        # elif skeleton_mode == 'heatmap':
        #     tbar = tqdm(pids)
        #     for pid in tbar:
        #         # import pdb;pdb.set_trace()
        #         cur_track_path = os.path.join(self.skeleton_path, pid, 'heatmaps.pkl')
        #         with open(cur_track_path, 'rb') as f:
        #             track = pickle.load(f)  # shape: (track_len, 17, 96, 72)
        #         skeleton_tracks.append(track)
        print('------------Done loading skeleton data-----------')
        return skeleton_tracks

    def max_min_bbox_size(self):
        obs_bboxes = np.array(self.samples['obs_bbox'])  # ltrb
        pred_bboxes = np.array(self.samples['pred_bbox'])
        bboxes = np.concatenate([obs_bboxes, pred_bboxes], axis=1)
        h = bboxes[:, :, 3] - bboxes[:, :, 1]
        w = bboxes[:, :, 2] - bboxes[:, :, 0]

        # PIE: 688, 375, 10, 3, 124, 40, 301, 109
        # JAAD: 635 329 18 7 101 43 296 129
        return np.max(h), np.max(w), np.min(h), np.min(w), np.percentile(h, 50), np.percentile(w, 50), np.percentile(h, 90), np.percentile(w, 90)
    
    def rm_small_bb(self, data, min_size):
        print('----------Remove small bb-----------')
        min_w, min_h = min_size
        idx = list(range(self.num_samples))
        new_idx = list(range(self.num_samples))

        bboxes = np.array(data['obs_bbox'])  # ltrb
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

    def rm_occluded(self, data, max_occ):
        print('----------Remove occluded---------')
        idx = list(range(self.num_samples))
        new_idx = list(range(self.num_samples))

        for i in idx:
            for occ in data['obs_occ'][i]:
                if occ > max_occ:
                    new_idx.remove(i)
                    break
        
        for k in data.keys():
            data[k] = data[k][new_idx]
        print('n samples before removing occluded', self.num_samples)
        self.num_samples = len(new_idx)
        print('n samples after removing occluded', self.num_samples)

        return data
    
    def rm_720x1280(self, all_data):
        # video to remove: 61~70
        rm_list = ['video_00'+str(i) for i in range(61, 71)]
        n_track = len(all_data['image'])
        idxs = list(range(n_track))
        new_idxs = list(range(n_track))
        print('Removing video 61~70')
        print(f'Num video before removing: {n_track}')
        for i in idxs:
            video = all_data['image'][i][0].split('/')[-2]
            vid = int(video.replace('video_', ''))
            if vid >= 61 and vid <= 70:
                new_idxs.remove(i)
        all_data.pop('image_dimension', None)
        for k in all_data:
            # print(np.array(all_data[k]).shape, k)
            all_data[k] = np.array(all_data[k])[new_idxs]
        print('Num video after removing: ', len(all_data['image']))
        return all_data


    def balance(self, balance_label, all_samples, random_seed=42):
        for lbl in all_samples[balance_label]:
            for i in lbl:  # 每条track
                if i not in [0, 1]:
                    raise Exception("The label values used for balancing must be"
                                    " either 0 or 1")
        print('---------------------------------------------------------')
        print("Balancing the number of positive and negative intention samples")
        gt_labels = [gt[0] for gt in all_samples[balance_label]]  # 每条track一个值
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples
        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
            return all_samples
        else:
            print('Unbalanced: \t Positive: {} \t \
                  Negative: {}'.format(num_pos_samples, num_neg_samples))
            if num_neg_samples > num_pos_samples:
                rm_index = np.where(np.array(gt_labels) == 0)[0]
                n_rm_index = np.where(np.array(gt_labels) == 1)[0]
                num_keep = num_pos_samples
            else:
                rm_index = np.where(np.array(gt_labels) == 1)[0]
                n_rm_index = np.where(np.array(gt_labels) == 0)[0]
                num_keep = num_neg_samples
            np.random.seed(random_seed)
            np.random.shuffle(rm_index)
            kp_index = rm_index[:num_keep]
            new_index = list(set(n_rm_index) | set(kp_index))

            for k in all_samples.keys():
                all_samples[k] = all_samples[k][new_index]
                assert len(all_samples[k]) == 2 * num_keep
            return all_samples


def save_cropped_imgs(resize_mode='even_padded', 
                      target_size=(224, 224), 
                      dataset_name='PIE', 
                      ):
    import os
    if dataset_name == 'PIE':
        cropped_root = '/home/y_feng/workspace6/datasets/PIE_dataset/cropped_images'
        dataset_root = '/home/y_feng/workspace6/datasets/PIE_dataset'
        data_base = PIE(data_path=dataset_root)
        data_opts = {'normalize_bbox': False,
                         'fstride': 1,
                         'sample_type': 'all',
                         'height_rng': [0, float('inf')],
                         'squarify_ratio': 0,
                         'data_split_type': 'default',  # kfold, random, default. default: set03 for test
                         'seq_type': 'intention',  # crossing , intention
                         'min_track_size': 0,  # discard tracks that are shorter
                         'max_size_observe': 1,  # number of observation frames
                         'max_size_predict': 1,  # number of prediction frames
                         'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
                         'balance': False,  # balance the training and testing samples
                         'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
                         'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
                         'seq_type': 'trajectory',
                         'encoder_input_type': ['bbox', 'obd_speed'],
                         'decoder_input_type': [],
                         'output_type': ['intention_binary', 'bbox']
                         }
    else:
        cropped_root = '/home/y_feng/workspace6/datasets/JAAD/cropped_images'
        dataset_root = '/home/y_feng/workspace6/datasets/JAAD'
        data_opts = {'fstride': 1,
             'sample_type': 'all',  
	         'subset': 'high_visibility',
             'data_split_type': 'default',
             'seq_type': 'trajectory',
	         'height_rng': [0, float('inf')],
	         'squarify_ratio': 0,
             'min_track_size': 0,
             'random_params': {'ratios': None,
                               'val_data': True,
                               'regen_data': True},
             'kfold_params': {'num_folds': 5, 'fold': 1}}
        data_base = JAAD(data_path='/home/y_feng/workspace6/datasets/JAAD')
    data_dir = os.path.join(cropped_root, resize_mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h')
    makedir(data_dir)

    tracks = data_base.generate_data_trajectory_sequence(image_set='all', 
                                                         **data_opts)  # all: 1842  train 882 
    # 'image', 'ped_id', 'bbox', 'center', 'occlusion', 'obd_speed', 'gps_speed', 'heading_angle', 'gps_coord', 'yrp', 'intention_prob', 'intention_binary'
    num_tracks = len(tracks['image'])
    # ids = []
    # for track in tracks['ped_id']:
    #     ids += track
    # id_set = np.unique(ids)
    # print(len(id_set))

    for i in range(num_tracks):
        cur_pid = tracks['ped_id'][i][0][0]  # [[id], [id], ...]
        ped_dir = os.path.join(data_dir, cur_pid)
        makedir(ped_dir)
        track_len = len(tracks['ped_id'][i])
        for j in range(track_len):
            img_path = tracks['image'][i][j]
            target_path = os.path.join(ped_dir, img_path.split('/')[-1])
            img = cv2.imread(img_path)
            l, t, r, b = tracks['bbox'][i][j]  # l t r b
            l, t, r, b = map(int, [l, t, r, b])
            cropped = img[t:b, l:r]
            if resize_mode == 'ori':
                resized = cropped
            elif resize_mode == 'resized':
                resized = cv2.resize(cropped, target_size)
            elif resize_mode == 'even_padded':
                h = b-t
                w = r-l
                if  float(w) / h > float(target_size[0]) / target_size[1]:
                    ratio = float(target_size[0]) / w
                else:
                    ratio = float(target_size[1]) / h
                new_size = (int(w*ratio), int(h*ratio))
                cropped = cv2.resize(cropped, new_size)
                w_pad = target_size[0] - new_size[0]
                h_pad = target_size[1] - new_size[1]
                l_pad = w_pad // 2
                r_pad = w_pad - l_pad
                t_pad = h_pad // 2
                b_pad = h_pad - t_pad
                resized = cv2.copyMakeBorder(cropped,
                                             t_pad,b_pad,l_pad,r_pad,
                                             cv2.BORDER_CONSTANT,
                                             value=(0, 0, 0))  # t, b, l, r
                assert (resized.shape[1], resized.shape[0]) == target_size
            else:
                h = b-t
                w = r-l
                if  float(w) / h > float(target_size[0]) / target_size[1]:
                    ratio = float(target_size[0]) / w
                else:
                    ratio = float(target_size[1]) / h
                new_size = (int(w*ratio), int(h*ratio))
                cropped = cv2.resize(cropped, new_size)
                w_pad = target_size[0] - new_size[0]
                h_pad = target_size[1] - new_size[1]
                resized = cv2.copyMakeBorder(cropped,0,h_pad,0,w_pad,cv2.BORDER_CONSTANT,value=(0, 0, 0))  # t, b, l, r
                assert (resized.shape[1], resized.shape[0]) == target_size
            cv2.imwrite(target_path, resized)
        print(i, ped_dir, 'done')

def save_context_imgs(mode='mask_ped', target_size=(224, 224), dataset_name='PIE'):
    import os
    if dataset_name == 'PIE':
        context_root = '/home/y_feng/workspace6/datasets/PIE_dataset/context'
        dataset_root = '/home/y_feng/workspace6/datasets/PIE_dataset'
        data_base = PIE(data_path=dataset_root)
        data_opts = {'normalize_bbox': False,
                         'fstride': 1,
                         'sample_type': 'all',
                         'height_rng': [0, float('inf')],
                         'squarify_ratio': 0,
                         'data_split_type': 'default',  # kfold, random, default. default: set03 for test
                         'seq_type': 'intention',  # crossing , intention
                         'min_track_size': 0,  # discard tracks that are shorter
                         'max_size_observe': 1,  # number of observation frames
                         'max_size_predict': 1,  # number of prediction frames
                         'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
                         'balance': False,  # balance the training and testing samples
                         'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
                         'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
                         'seq_type': 'trajectory',
                         'encoder_input_type': ['bbox', 'obd_speed'],
                         'decoder_input_type': [],
                         'output_type': ['intention_binary', 'bbox']
                         }
    else:
        context_root = '/home/y_feng/workspace6/datasets/JAAD/context'
        dataset_root = '/home/y_feng/workspace6/datasets/JAAD'
        data_opts = {'fstride': 1,
             'sample_type': 'all',  
	         'subset': 'high_visibility',
             'data_split_type': 'default',
             'seq_type': 'trajectory',
	         'height_rng': [0, float('inf')],
	         'squarify_ratio': 0,
             'min_track_size': 0,
             'random_params': {'ratios': None,
                               'val_data': True,
                               'regen_data': True},
             'kfold_params': {'num_folds': 5, 'fold': 1}}
        data_base = JAAD(data_path=dataset_root)
    makedir(context_root)
    data_dir = os.path.join(context_root, mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h')
    makedir(data_dir)

    tracks = data_base.generate_data_trajectory_sequence(image_set='all', **data_opts)
    num_tracks = len(tracks['image'])
    mask_value = (127, 127, 127)

    if mode == 'mask_ped':
        for i in range(num_tracks):
            cur_pid = tracks['ped_id'][i][0][0]  # [[id], [id], ...]
            ped_dir = os.path.join(data_dir, cur_pid)
            makedir(ped_dir)
            track_len = len(tracks['ped_id'][i])
            for j in range(track_len):
                img_path = tracks['image'][i][j]
                target_path = os.path.join(ped_dir, img_path.split('/')[-1])
                img = cv2.imread(img_path)
                l, t, r, b = tracks['bbox'][i][j]  # l t r b
                l, t, r, b = map(int, [l, t, r, b])
                # if mode == 'mask_ped':
                rect = np.array([[l, t], [r, t], [r, b], [l, b]])
                masked = cv2.fillConvexPoly(img, rect, mask_value)
                resized = cv2.resize(masked, target_size)
                cv2.imwrite(target_path, resized)
            print(i, ped_dir, 'done')

    elif mode == 'ori':
        for i in range(num_tracks):
            cur_pid = tracks['ped_id'][i][0][0]  # [[id], [id], ...]
            ped_dir = os.path.join(data_dir, cur_pid)
            makedir(ped_dir)
            track_len = len(tracks['ped_id'][i])
            for j in range(track_len):
                img_path = tracks['image'][i][j]
                target_path = os.path.join(ped_dir, img_path.split('/')[-1])
                img = cv2.imread(img_path)
                resized = cv2.resize(img, target_size)
                cv2.imwrite(target_path, resized)
            print(i, ped_dir, 'done')

    elif mode == 'ori_local':
        for i in range(num_tracks):
            cur_pid = tracks['ped_id'][i][0][0]  # [[id], [id], ...]
            ped_dir = os.path.join(data_dir, cur_pid)
            makedir(ped_dir)
            track_len = len(tracks['ped_id'][i])
            for j in range(track_len):
                img_path = tracks['image'][i][j]
                target_path = os.path.join(ped_dir, img_path.split('/')[-1])
                img = cv2.imread(img_path)
                ori_H, ori_W = img.shape[0], img.shape[1]
                l, t, r, b = tracks['bbox'][i][j]  # l t r b
                l, t, r, b = map(int, [l, t, r, b])
                # crop local context
                x = (l+r) // 2
                y = (t+b) // 2
                h = b-t
                w = r-l
                crop_h = h*2
                crop_w = h*2
                crop_l = max(x-h, 0)
                crop_r = min(x+h, ori_W)
                crop_t = max(y-h, 0)
                crop_b = min(y+h, ori_W)
                cropped = img[crop_t:crop_b, crop_l:crop_r]
                l_pad = max(h-x, 0)
                r_pad = max(x+h-ori_W, 0)
                t_pad = max(h-y, 0)
                b_pad = max(y+h-ori_H, 0)
                cropped = cv2.copyMakeBorder(cropped, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, value=mask_value)
                assert cropped.shape[0] == crop_h and cropped.shape[1] == crop_w, (cropped.shape, (crop_h, crop_w))
                resized = cv2.resize(cropped, target_size)
                cv2.imwrite(target_path, resized)
            print(i, ped_dir, 'done')
    elif mode == 'local':
        for i in range(num_tracks):
            cur_pid = tracks['ped_id'][i][0][0]  # [[id], [id], ...]
            ped_dir = os.path.join(data_dir, cur_pid)
            makedir(ped_dir)
            track_len = len(tracks['ped_id'][i])
            for j in range(track_len):
                img_path = tracks['image'][i][j]
                target_path = os.path.join(ped_dir, img_path.split('/')[-1])
                img = cv2.imread(img_path)
                ori_H, ori_W = img.shape[0], img.shape[1]
                l, t, r, b = tracks['bbox'][i][j]  # l t r b
                l, t, r, b = map(int, [l, t, r, b])
                # mask target pedestrian
                rect = np.array([[l, t], [r, t], [r, b], [l, b]])
                masked = cv2.fillConvexPoly(img, rect, mask_value)
                # crop local context
                x = (l+r) // 2
                y = (t+b) // 2
                h = b-t
                w = r-l
                crop_h = h*2
                crop_w = h*2
                crop_l = max(x-h, 0)
                crop_r = min(x+h, ori_W)
                crop_t = max(y-h, 0)
                crop_b = min(y+h, ori_W)
                cropped = masked[crop_t:crop_b, crop_l:crop_r]
                l_pad = max(h-x, 0)
                r_pad = max(x+h-ori_W, 0)
                t_pad = max(h-y, 0)
                b_pad = max(y+h-ori_H, 0)
                cropped = cv2.copyMakeBorder(cropped, 
                                             t_pad, b_pad, l_pad, r_pad, 
                                             cv2.BORDER_CONSTANT, 
                                             value=mask_value)
                assert cropped.shape[0] == crop_h and \
                    cropped.shape[1] == crop_w, \
                    (cropped.shape, (crop_h, crop_w))
                resized = cv2.resize(cropped, target_size)
                cv2.imwrite(target_path, resized)
            print(i, ped_dir, 'done')

def track_vehicles():
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='crop')

    parser.add_argument('--h', type=int, default=224)
    parser.add_argument('--w', type=int, default=224)
    parser.add_argument('--dataset_name', type=str, default='PIE')
    # crop args
    parser.add_argument('--resize_mode', type=str, default='padded')
    # context args
    parser.add_argument('--context_mode', type=str, default='mask_ped')

    args = parser.parse_args()

    if args.action == 'crop':
        save_cropped_imgs(resize_mode=args.resize_mode, 
                          target_size=(args.w, args.h), 
                          dataset_name=args.dataset_name)
    elif args.action == 'context':
        save_context_imgs(mode=args.context_mode, 
                          target_size=(args.w, args.h), 
                          dataset_name=args.dataset_name)
    elif args.action == 'pseudo_heatmap':
        coord2pseudo_heatmap(h=args.h, 
                             w=args.w, 
                             dataset_name=args.dataset_name)