import os
import json
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import pickle
import copy
from tqdm import tqdm
import pdb
import torch
import numpy as np
from scipy import interpolate
from torchvision.transforms import functional as TVF
from ..data.preprocess import bdd100k_get_vidnm2vidid
from ..data.normalize import img_mean_std_BGR, norm_imgs, sklt_local_to_global, norm_bbox, norm_sklt
from ..data.transforms import RandomHorizontalFlip, RandomResizedCrop, crop_local_ctx
from ..data.bbox import bbox2d_relation_multi_seq, pad_neighbor
from .dataset_id import DATASET_TO_ID, ID_TO_DATASET

from config import dataset_root

RM_VID_NMS = ['036aea46-ee63a8e7', 
              '018aca44-9a616a49', 
              '0059f17f-f0882eef', 
              '03d75f61-ce25863a', 
              '0033b19f-65613f7e', 
              '0062298d-fd69d0ec', 
              '034daca2-a038e325', 
              '009aecce-ce4a9413', 
              '0001542f-7c670be8', 
              '028357c9-f08394e7', 
              '00cea101-06f21d5e', 
              '030e8b7a-4dab7745', 
              '01118704-2d838d7f', 
              '0062298d-cbbec2cd', 
              '01c2e726-c3a655b2', 
              '023d0f3c-564e6d31', 
              '009e7a6a-3d755b8b', 
              '02514ff2-b8e3551c', 
              '0062298d-2d787502', 
              '0062298d-e6abad2f', 
              '00cb28b9-08a22af7', 
              '0342543b-cb4084bd', 
              '01c2e726-414a03ea', 
              'b2102d00-a8c09be1', 
              'b20b69d2-6e2b9e73', 
              'b1f4491b-bf7d513f',
              'b1f4491b-09593e90']

# obj id (unique)
# vid id --> img id
class BDD100kDataset(torch.utils.data.Dataset):
    def __init__(self,
                 subsets='train_val',
                 dataset_root=dataset_root,
                 obs_len=4, pred_len=4, overlap_ratio=0.5,
                 offset_traj=False,
                 obs_fps=2,
                 target_color_order='BGR', img_norm_mode='torch',
                 small_set=0,
                 min_h=72,
                 min_w=36,
                 resize_mode='even_padded', crop_size=(224, 224),
                 modalities=['img', 'sklt', 'ctx', 'traj', 'ego', 'social'],
                 img_format='',
                 ctx_format='ped_graph', ctx_size=(224, 224),
                 sklt_format='pseudo_heatmap',
                 traj_format='ltrb',
                 ego_format='accel',
                 social_format='rel_loc',
                 augment_mode='random_hflip',
                 seg_cls=['person', 'vehicle', 'road', 'traffic_light'],
                 rm_occl=False,
                 rm_trun=False,
                 tte=None,
                 max_n_neighbor=10,
                 min_wh=(72,36),
                 ):
        super().__init__()
        self.dataset_name = 'bdd100k'
        print(f'------------------Init{self.dataset_name}------------------')
        self.img_size = (720, 1280)
        self.fps = 5
        self.data_root = os.path.join(dataset_root, 'BDD100k/bdd100k')
        self.fps = 5
        self.subsets = subsets
        if self.subsets == 'test':
            self.subsets = 'val'
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_interval = self.fps // obs_fps - 1
        self.overlap_ratio = overlap_ratio
        self.norm_traj = offset_traj
        self.target_color_order = target_color_order
        self.img_norm_mode = img_norm_mode
        self.img_mean, self.img_std = img_mean_std_BGR(self.img_norm_mode)
        # sequence length considering interval
        self.min_h = min_h
        self.min_w = min_w
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
        self.small_set = small_set
        self.augment_mode = augment_mode
        self.seg_cls = seg_cls
        self.rm_occl = rm_occl
        self.rm_trun = rm_trun
        self.tte = tte
        self.max_n_neighbor = max_n_neighbor

        self.transforms = {'random': 0,
                            'balance': 0,
                            'hflip': None,
                            'resized_crop': {'img': None,
                                            'ctx': None,
                                            'sklt': None}}
        self.subsets = self.subsets.split('_')
        self.extra_root = os.path.join(self.data_root, 
                                       'extra')
        # get dict
        vid_nm2id_path = os.path.join(self.extra_root, 
                                      'vid_nm2id.pkl')
        vid_id2nm_path = os.path.join(self.extra_root, 
                                      'vid_id2nm.pkl')
        if os.path.exists(vid_nm2id_path) and \
            os.path.exists(vid_id2nm_path):
            with open(vid_id2nm_path, 'rb') as f:
                self.vid_id2nm = pickle.load(f)
            with open(vid_nm2id_path, 'rb') as f:
                self.vid_nm2id = pickle.load(f)
        else:
            # id: int  nm: str
            self.vid_id2nm, self.vid_nm2id = \
                bdd100k_get_vidnm2vidid(data_root=self.data_root,
                                        sub_set=self.subsets)
        self.imgnm_to_objid_path = os.path.join(self.extra_root, 
                                                'imgnm_to_objid_to_ann.pkl')
        
        # get tracks
        self.p_tracks, self.v_tracks = self.get_tracks()
        # add the acceleration to the pedestrian tracks
        self.p_tracks = self._get_accel(self.p_tracks)
        self.v_tracks = self._get_accel(self.v_tracks)

        # get neighbor info
        if os.path.exists(self.imgnm_to_objid_path) and False:
            with open(self.imgnm_to_objid_path, 'rb') as f:
                self.imgnm_to_objid = pickle.load(f)
        else:
            self.imgnm_to_objid = \
                self.get_imgnm_to_objid(self.p_tracks, 
                                        self.v_tracks, 
                                        self.imgnm_to_objid_path)
        # convert tracks into samples
        self.samples = self.tracks_to_samples(self.p_tracks)

        # get num samples
        self.num_samples = len(self.samples['obs']['img_id'])

        # get neighbors
        if 'social' in self.modalities:
            self.neighbor_seq_path = os.path.join(self.extra_root,
                f'fps_{self.fps}_obs_{self.obs_len}_pred_{self.pred_len}_interval_{self.seq_interval}_overlap_{self.overlap_ratio}.pkl')
            self.samples = self.get_neighbor_relation(self.samples,
                                                self.neighbor_seq_path)

        # apply interval
        if self.seq_interval > 0:
            self.downsample_seq()
        # add augmentation transforms
        self._add_augment(self.samples)
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

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        obs_bbox_offset = copy.deepcopy(torch.tensor(self.samples['obs']['bbox_normed'][idx]).float())
        obs_bbox = copy.deepcopy(torch.tensor(self.samples['obs']['bbox'][idx]).float())
        pred_bbox_offset = copy.deepcopy(torch.tensor(self.samples['pred']['bbox_normed'][idx]).float())
        pred_bbox = copy.deepcopy(torch.tensor(self.samples['pred']['bbox'][idx]).float())
        obs_ego = copy.deepcopy(torch.tensor(self.samples['obs']['ego_motion'][idx]).float().unsqueeze(1))
        assert len(obs_ego.shape) == 2
        vid_id_int = torch.tensor(int(self.samples['obs']['vid_id'][idx][0]))
        obj_id_int = torch.tensor(int(float(self.samples['obs']['obj_id'][idx][0])))
        img_id_int = torch.tensor(self.samples['obs']['img_id_int'][idx])
        obs_bbox_ori = copy.deepcopy(obs_bbox)
        pred_bbox_ori = copy.deepcopy(pred_bbox)
        # squeeze the coords
        if '0-1' in self.traj_format:
            obs_bbox_offset = norm_bbox(obs_bbox_offset, self.dataset_name)
            pred_bbox_offset = norm_bbox(pred_bbox_offset, self.dataset_name)
            obs_bbox = norm_bbox(obs_bbox, self.dataset_name)
            pred_bbox = norm_bbox(pred_bbox, self.dataset_name)
        sample = {'dataset_name': torch.tensor(DATASET_TO_ID[self.dataset_name]),
                  'set_id_int': torch.tensor(-1),
                  'vid_id_int': vid_id_int,  # int
                  'ped_id_int': obj_id_int,  # int
                  'img_nm_int': img_id_int,
                  'obs_bboxes': obs_bbox_offset,
                  'obs_bboxes_unnormed': obs_bbox,
                  'obs_bboxes_ori': obs_bbox_ori,
                  'obs_ego': obs_ego,
                  'pred_act': torch.tensor(-1),
                  'pred_bboxes': pred_bbox_offset,
                  'pred_bboxes_unnormed': pred_bbox,
                  'pred_bboxes_ori': pred_bbox_ori,
                  'atomic_actions': torch.tensor(-1),
                  'simple_context': torch.tensor(-1),
                  'complex_context': torch.tensor(-1),  # (1,)
                  'communicative': torch.tensor(-1),
                  'transporting': torch.tensor(-1),
                  'age': torch.tensor(-1),
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
                  pad_neighbor([copy.deepcopy(np.array(self.samples['obs']['neighbor_relations'][idx]).transpose(1,0,2)),
                                copy.deepcopy(np.array(self.samples['obs']['neighbor_bbox'][idx]).transpose(1,0,2)),
                                copy.deepcopy(self.samples['obs']['neighbor_oid'][idx])
                                ],
                                self.max_n_neighbor)
            if self.social_format == 'rel_loc':
                sample['obs_neighbor_relation'] = torch.tensor(relations).float()  # K T 5
            elif self.social_format == 'ori_traj':
                sample['obs_neighbor_relation'] =\
                      torch.cat([obs_bbox_ori.unsqueeze(0), torch.tensor(neighbor_bbox).float()], 0) # K+1 T 4
            sample['obs_neighbor_bbox'] = torch.tensor(neighbor_bbox).float()
            sample['obs_neighbor_oid'] = torch.tensor(neighbor_oid).float()
        if 'img' in self.modalities:
            imgs = []
            for img_id in self.samples['obs']['img_id_int'][idx]:
                img_path = os.path.join(self.extra_root,
                                        'cropped_images',
                                        self.resize_mode,
                                        '224w_by_224h',
                                        'ped',
                                        str(self.samples['obs']['obj_id'][idx][0]),
                                        str(img_id)+'.png'
                                        )
                imgs.append(cv2.imread(img_path))
            imgs = np.stack(imgs, axis=0)
            # (T, H, W, C) -> (C, T, H, W)if self.sklt_format == 'pseudo_heatmap':
            ped_imgs = torch.from_numpy(imgs).float().permute(3, 0, 1, 2)
            # normalize img
            if self.img_norm_mode != 'ori':
                ped_imgs = norm_imgs(ped_imgs, self.img_mean, self.img_std)
            # BGR -> RGB
            if self.target_color_order == 'RGB':
                ped_imgs = torch.flip(ped_imgs, dims=[0])
            sample['ped_imgs'] = ped_imgs
        if 'sklt' in self.modalities:
            interm_dir = 'even_padded/48w_by_48h' \
                if self.sklt_format == 'pseudo_heatmap' else 'even_padded/288w_by_384h'
            sklts = []
            for img_id in self.samples['obs']['img_id_int'][idx]:
                sklt_path = os.path.join(self.extra_root,
                                        'sk_'+self.sklt_format.replace('0-1', '')+'s',
                                        interm_dir,
                                        str(self.samples['obs']['obj_id'][idx][0]),
                                        str(img_id)+'.pkl'
                                        )
                with open(sklt_path, 'rb') as f:
                    heatmap = pickle.load(f)
                sklts.append(heatmap)
            sklts = np.stack(sklts, axis=0)  # T, ...
            pred_sklts = []
            for img_id in self.samples['pred']['img_id_int'][idx]:
                sklt_path = os.path.join(self.extra_root,
                                        'sk_'+self.sklt_format.replace('0-1', '')+'s',
                                        interm_dir,
                                        str(self.samples['pred']['obj_id'][idx][0]),
                                        str(img_id)+'.pkl'
                                        )
                with open(sklt_path, 'rb') as f:
                    heatmap = pickle.load(f)
                pred_sklts.append(heatmap)
            pred_sklts = np.stack(pred_sklts, axis=0)  # T, ...
            if 'coord' in self.sklt_format:
                obs_skeletons = torch.from_numpy(sklts).float().permute(2, 0, 1)[:2]  # shape: (2, T, nj)
                pred_skeletons = torch.from_numpy(pred_sklts).float().permute(2, 0, 1)[:2] # (2, T, nj)
                # yx -> xy
                obs_skeletons = obs_skeletons.flip(0)
                pred_skeletons = pred_skeletons.flip(0)
                # add offset
                obs_skeletons = sklt_local_to_global(obs_skeletons.float(), obs_bbox_ori.float())
                pred_skeletons = sklt_local_to_global(pred_skeletons.float(), pred_bbox_ori.float())
                if '0-1' in self.sklt_format:
                    obs_skeletons[0] = obs_skeletons[0] / self.img_size[0]
                    obs_skeletons[1] = obs_skeletons[1] / self.img_size[1]
                    pred_skeletons[0] = pred_skeletons[0] / self.img_size[0]
                    pred_skeletons[1] = pred_skeletons[1] / self.img_size[1]
            elif self.sklt_format == 'pseudo_heatmap':
                # T C H W -> C T H W
                obs_skeletons = torch.from_numpy(sklts).float().permute(1, 0, 2, 3)  # shape: (17, seq_len, 48, 48)
                pred_skeletons = torch.from_numpy(pred_sklts).float().permute(1, 0, 2, 3)
            sample['obs_skeletons'] = obs_skeletons
            sample['pred_skeletons'] = pred_skeletons

        if 'ctx' in self.modalities:
            if self.ctx_format in ('local', 'ori_local', 'mask_ped', 'ori',
                                    'ped_graph', 'ped_graph_seg'):
                ctx_imgs = []
                if self.ctx_format in ('ped_graph', 'ped_graph_seg'):
                    ctx_format_dir = 'ori_local'
                else:
                    ctx_format_dir = self.ctx_format
                for img_id in self.samples['obs']['img_id_int'][idx]:
                    img_path = os.path.join(self.extra_root,
                                        'context',
                                        ctx_format_dir,
                                        '224w_by_224h',
                                        'ped',
                                        str(self.samples['obs']['obj_id'][idx][0]),
                                        str(img_id)+'.png'
                                        )
                    ctx_imgs.append(cv2.imread(img_path))
                ctx_imgs = np.stack(ctx_imgs, axis=0)
                # (T, H, W, C) -> (C, T, H, W)
                ctx_imgs = torch.from_numpy(ctx_imgs).float().permute(3, 0, 1, 2)
                # normalize img
                if self.img_norm_mode != 'ori':
                    ctx_imgs = norm_imgs(ctx_imgs, 
                                         self.img_mean, self.img_std)
                # BGR -> RGB
                if self.target_color_order == 'RGB':
                    ctx_imgs = torch.flip(ctx_imgs, dims=[0])
                # load cropped segmentation map
                if self.ctx_format in ('ped_graph', 'ped_graph_seg'):
                    all_c_seg = []
                    oid = self.samples['obs']['obj_id'][idx][0]
                    img_id = self.samples['obs']['img_id_int'][idx][-1]
                    for c in self.seg_cls:
                        seg_path = os.path.join(self.data_root,
                                                'extra/cropped_seg',
                                                c,
                                                'ori_local/224w_by_224h',
                                                'ped',
                                                str(oid),
                                                str(img_id)+'.pkl')
                        try:
                            with open(seg_path, 'rb') as f:
                                segmap = pickle.load(f)*1  # h w int
                        except:
                            import pdb; pdb.set_trace()
                        all_c_seg.append(torch.from_numpy(segmap))
                    all_c_seg = torch.stack(all_c_seg, dim=-1)  # h w n_cls
                    all_c_seg = torch.argmax(all_c_seg, dim=-1, keepdim=True).permute(2, 0, 1)  # 1 h w
                    ctx_imgs = torch.concat([ctx_imgs[:, -1], all_c_seg], dim=0).unsqueeze(1)  # 4 1 h w

                sample['obs_context'] = ctx_imgs  # shape [3(or 4), obs_len, H, W]
            elif self.ctx_format in \
                ('seg_ori_local', 'seg_local'):
                ctx_imgs = []
                for img_id in self.samples['obs']['img_id_int'][idx]:
                    img_path = os.path.join(self.extra_root,
                                        'context',
                                        'ori_local',
                                        '224w_by_224h',
                                        'ped',
                                        str(self.samples['obs']['obj_id'][idx][0]),
                                        str(img_id)+'.png'
                                        )
                    ctx_imgs.append(cv2.imread(img_path))
                ctx_imgs = np.stack(ctx_imgs, axis=0)
                # (T, H, W, C) -> (C, T, H, W)
                ctx_imgs = torch.from_numpy(ctx_imgs).float().permute(3, 0, 1, 2)
                # normalize img
                if self.img_norm_mode != 'ori':
                    ctx_imgs = norm_imgs(ctx_imgs, self.img_mean, self.img_std)
                # BGR -> RGB
                if self.target_color_order == 'RGB':
                    ctx_imgs = torch.flip(ctx_imgs, dims=[0])  # 3THW
                # load segs
                ctx_segs = {c:[] for c in self.seg_cls}
                for c in self.seg_cls:
                    for img_id in self.samples['obs']['img_id_int'][idx]:
                        seg_path = os.path.join(
                            self.extra_root,
                            'seg_sam',
                            c,
                            str(self.samples['obs']['vid_id'][idx][0]),
                            str(img_id)+'.pkl'
                        )
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
                if self.ctx_format == 'ped_graph':
                    all_seg = torch.argmax(all_seg[0, -1], dim=-1, keepdim=True).permute(2, 0, 1)  # 1 h w
                    sample['obs_context'] = torch.concat([ctx_imgs[:, -1], all_seg], dim=0)
                else:
                    sample['obs_context'] = all_seg * torch.unsqueeze(ctx_imgs, dim=-1)  # 3Thw n_cls

        # augmentation
        if self.augment_mode != 'none':
            if self.transforms['random']:
                sample = self._random_augment(sample)
        
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
        return sample
    
    def get_tracks(self):
        # p_tracks = {'vid_id': [],
        #             'img_nm': [],
        #             'img_nm_int': [],
        #             'obj_id': [],
        #             'bbox': [],
        #             # 'motion_status': [],
        #             'ego_motion': []}
        track_dict = {
            'ped':{},  # {oid: {vid_id:[], img_nm:[[]], img_nm_int:[[]], bbox:[[]], ego_motion:[[]]}}
            'veh':{},
        }
        label_root = os.path.join(self.data_root, 'labels/box_track_20')
        ego_info_root = os.path.join(self.data_root, 'info/100k')
        print('Getting tracks')
        for _subset in tqdm(self.subsets, ascii=True,desc='subset loop'):
            l_subset_dir = os.path.join(label_root, _subset)
            # remove videos without enough ego info
            _f_nm_list = os.listdir(l_subset_dir)
            f_nms_to_rm = [nm+'.json' for nm in RM_VID_NMS]
            f_nm_list = list(set(_f_nm_list) - set(f_nms_to_rm))
            # traverse video
            for f_nm in tqdm(f_nm_list, ascii=True,desc='vid loop'):
                label_f_path = os.path.join(l_subset_dir, f_nm)
                ego_f_path = os.path.join(ego_info_root, _subset, f_nm)
                with open(label_f_path) as f:
                    vid_label = json.load(f)
                with open(ego_f_path) as f:
                    vid_ego = json.load(f)
                loc_list = vid_ego['locations']
                max_num_sec = len(loc_list)
                spd_list = [s['speed'] for s in loc_list]
                # interpolate the speed seq
                new_x = np.linspace(0, 
                                    max_num_sec-1, 
                                    max_num_sec*5-4)
                spd_list_interp = np.interp(new_x, 
                                            np.arange(max_num_sec), 
                                            spd_list)
                vid_nm = f_nm.replace('.json', '')
                vid_id = self.vid_nm2id[vid_nm]
                # traverse img label
                for i in range(min(len(spd_list_interp), len(vid_label))):
                    img_label = vid_label[i]
                    spd = spd_list_interp[i]
                    if len(img_label['labels']) == 0:
                        continue
                    img_nm = img_label['name'].split('-')[-1]
                    img_id = img_nm.replace('.jpg', '')
                    img_id_int = int(img_id)
                    for l in img_label['labels']:
                        occl = l['attributes']['occluded']
                        trun = l['attributes']['truncated']
                        ltrb = [l['box2d']['x1'],
                                l['box2d']['y1'],
                                l['box2d']['x2'],
                                l['box2d']['y2']]
                        if self.rm_occl and occl \
                            or self.rm_trun and trun \
                            or ltrb[2]-ltrb[0]<self.min_w \
                            or ltrb[3]-ltrb[1]<self.min_h:
                            continue
                        oid_int = int(l['id'])
                        oid = str(oid_int)
                        cls = 'ped' if l['category'] in ('other person', 'pedestrian', 'rider') else 'veh'
                        
                        if oid not in track_dict[cls]:
                            track_dict[cls][oid] = {
                                'vid_id':[[vid_id]],
                                'img_id':[[str(img_id_int)]],
                                'img_id_int':[[img_id_int]],
                                'bbox':[[ltrb]],
                                'ego_speed':[[spd]],
                            }
                        else:
                            # check the continuity
                            if len(track_dict[cls][oid]['img_id_int'][-1]) > 0 \
                                and img_id_int - track_dict[cls][oid]['img_id_int'][-1][-1] > 1:
                                # init a new track
                                new_track = {
                                    'vid_id':[vid_id],
                                    'img_id':[str(img_id_int)],
                                    'img_id_int':[img_id_int],
                                    'bbox':[ltrb],
                                    'ego_speed':[spd],
                                }
                                for k in track_dict[cls][oid]:
                                    track_dict[cls][oid][k].append(new_track[k])
                            else:
                                track_dict[cls][oid]['vid_id'][-1].append(vid_id)
                                track_dict[cls][oid]['img_id'][-1].append(str(img_id_int))
                                track_dict[cls][oid]['img_id_int'][-1].append(img_id_int)
                                track_dict[cls][oid]['bbox'][-1].append(ltrb)
                                track_dict[cls][oid]['ego_speed'][-1].append(spd)
        p_tracks = {'vid_id':[],  # int
                    'obj_id':[],  # str
                    'img_id':[],  # str
                    'img_id_int':[],  # int
                    'bbox':[],
                    'ego_speed':[],
                    }
        v_tracks = {'vid_id':[],
                    'obj_id':[],
                    'img_id':[],
                    'img_id_int':[],
                    'bbox':[],
                    'ego_speed':[],
                    }
        for oid in track_dict['ped']:
            for i in range(len(track_dict['ped'][oid]['img_id_int'])):
                for k in track_dict['ped'][oid]:
                    p_tracks[k].append(track_dict['ped'][oid][k][i])
                p_tracks['obj_id'].append([oid for _ in track_dict['ped'][oid]['img_id_int'][i]])
        for oid in track_dict['veh']:
            for i in range(len(track_dict['veh'][oid]['img_id_int'])):
                for k in track_dict['veh'][oid]:
                    v_tracks[k].append(track_dict['veh'][oid][k][i])
                v_tracks['obj_id'].append([oid for _ in track_dict['veh'][oid]['img_id_int'][i]])
        p_tracks['ego_motion'] = copy.deepcopy(p_tracks['ego_speed'])
        v_tracks['ego_motion'] = copy.deepcopy(v_tracks['ego_speed'])
        # n_long = 0
        # n_short = 0
        # for oid in track_dict['ped']:
        #     for t in track_dict['ped'][oid]['img_id']:
        #         if len(t) > 1:
        #             n_long += 1
        #         else:
        #             n_short += 1
        # print(n_long, n_short)
        # pdb.set_trace()
        return p_tracks, v_tracks

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
                target_cid = samples['obs']['vid_id'][i][0]  # str
                target_oid = samples['obs']['obj_id'][i][0]  # str
                target_bbox_seq = np.array(samples['obs']['bbox'][i])  # T, 4
                obslen = len(target_bbox_seq)
                bbox_seq_dict = {'ped':{},
                                'veh':{}}
                for j in range(obslen):
                    imgnm = samples['obs']['img_id'][i][j]  # str
                    cur_ped_ids = set(self.imgnm_to_objid[target_cid][imgnm]['ped'].keys())
                    try:
                        cur_ped_ids.remove(target_oid)
                    except:
                        import pdb;pdb.set_trace()
                        raise ValueError()
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
            with open(save_path, 'wb') as f:
                pickle.dump(neighbor_seq, f)
        # add neighbor info to samples
        for k in neighbor_seq:
            samples['obs'][k] = neighbor_seq[k]
        return samples

    def get_imgnm_to_objid(self,
                           p_tracks, 
                           v_tracks, 
                           save_path):
        # vid id --> img nm --> obj type --> obj id -> bbox
        imgnm_to_oid_to_info = {}
        n_p_tracks = len(p_tracks['bbox'])
        print(f'Saving img_nm to obj_id dict of {self.dataset_name}')
        print('Pedestrian')
        for i in range(n_p_tracks):
            cid = p_tracks['vid_id'][i][0]
            oid = p_tracks['obj_id'][i][0]
            if cid not in imgnm_to_oid_to_info:
                imgnm_to_oid_to_info[cid] = {}
            for j in range(len(p_tracks['img_id'][i])):
                imgnm = p_tracks['img_id'][i][j]
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
            cid = v_tracks['vid_id'][i][0]
            oid = v_tracks['obj_id'][i][0]
            if cid not in imgnm_to_oid_to_info:
                imgnm_to_oid_to_info[cid] = {}
            for j in range(len(v_tracks['img_id'][i])):
                imgnm = v_tracks['img_id'][i][j]
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

    def _get_accel(self, obj_tracks):
        print('Getting ego acceleration')
        new_tracks = {}
        for k in obj_tracks:
            new_tracks[k] = []
        new_tracks['ego_accel'] = []
        # calculate acceleration and record the idx of tracks to remove
        idx_to_remove = []
        for i in range(len(obj_tracks['ego_speed'])):
            speed_track = obj_tracks['ego_speed'][i]
            # pdb.set_trace()
            if len(speed_track) < 2:
                idx_to_remove.append(i)
                continue
            new_tracks['ego_accel'].append([])
            for j in range(len(speed_track)-1):
                accel = (speed_track[j+1] - speed_track[j]) * self.fps  # 2 fps
                new_tracks['ego_accel'][-1].append(accel)
        # add the rest keys
        for k in obj_tracks:
            for i in range(len(obj_tracks[k])):
                if i in idx_to_remove:
                    continue
                cur_track = obj_tracks[k][i]
                new_tracks[k].append(cur_track[1:])

        # check if the lengths of all keys confront
        for k in new_tracks:
            assert len(new_tracks[k]) == len(new_tracks['ego_accel']), \
                (len(new_tracks[k]), len(new_tracks['ego_accel']))
            for i in range(len(new_tracks[k])):
                assert len(new_tracks[k][i]) == len(new_tracks['ego_accel'][i]), \
                    (len(new_tracks[k][i]), len(new_tracks['ego_accel'][i]))
        if 'accel' in self.ego_format:
            new_tracks['ego_motion'] = copy.deepcopy(new_tracks['ego_accel'])
        return new_tracks
    
    def tracks_to_samples(self, tracks):
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
                # skip too short sequences
                if len(track) < seq_len:
                    continue
                if self.tte is not None:
                    raise NotImplementedError()
                else:
                    _samples.extend([track[i: i+seq_len] \
                                        for i in range(0,
                                                    len(track)-seq_len+1, 
                                                    overlap_s)])
            samples[k] = _samples
        #  Normalize tracks by subtracting bbox/center at first time step from the rest
        print('---------------Normalize traj---------------')
        bbox_normed = copy.deepcopy(samples['bbox'])
        if self.norm_traj:
            for i in range(len(bbox_normed)):
                bbox_normed[i] = np.subtract(bbox_normed[i][:], bbox_normed[i][0]).tolist()
        samples['bbox_normed'] = bbox_normed
        # choose ego motion quantity
        
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
    
    def downsample_seq(self):
        for k in self.samples['obs']:
            if len(self.samples['obs'][k][0]) == self._obs_len\
                and (k not in ('neighbor_oid', 'neighbor_cls')):
                new_k = []
                for s in range(len(self.samples['obs'][k])):
                    ori_seq = self.samples['obs'][k][s]
                    new_seq = []
                    for i in range(0, self._obs_len, self.seq_interval+1):
                        try:
                            new_seq.append(ori_seq[i])
                        except:
                            import pdb;pdb.set_trace()
                            raise ValueError()
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
        


def check_ego_v_info():
    gps_root = '/home/y_feng/workspace6/datasets/BDD100k/bdd100k/info/100k'
    label_root = '/home/y_feng/workspace6/datasets/BDD100k/bdd100k/labels/box_track_20'
    cnt = 0
    for _subset in os.listdir(label_root):
        for f_nm in os.listdir(os.path.join(label_root, _subset)):
            if not os.path.exists(os.path.join(gps_root, _subset, f_nm)):
                cnt+=1
                print(f_nm)
    print(cnt)
    rm_vid_by_loc = []
    rm_vid_by_accel = []
    for _subset in os.listdir(label_root):
        for f_nm in os.listdir(os.path.join(label_root, _subset)):
            gps_f_path = os.path.join(os.path.join(gps_root, _subset, f_nm))
            with open(gps_f_path) as f:
                info = json.load(f)
            loc = info['locations']
            # try:
            #     gps = info['gps']
            # except KeyError:
            #     print(f_nm)
            #     print(info.keys())
            #     return
            # print(loc[1]['timestamp']-loc[0]['timestamp'], (loc[-1]['timestamp']-loc[0]['timestamp'])/len(loc))
            with open(os.path.join(label_root, _subset, f_nm)) as f:
                v_label = json.load(f)
            label_len = len(v_label)

            # if len(loc) < 40:
            #     print(f'location len {len(loc)} label len {len(v_label)}, {len(v_label)/5}')
            if len(loc) < label_len//5:
                rm_vid_by_loc.append(f_nm.replace('.json', ''))
                print(f'location len {len(loc)} label len {label_len}, {label_len//5}')
    print(rm_vid_by_loc)
    print(len(rm_vid_by_loc))
    

if __name__ == '__main__':
    check_ego_v_info()
    pass
