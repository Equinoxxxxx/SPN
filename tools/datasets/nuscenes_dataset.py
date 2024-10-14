from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from ..data.nusc_split import TRAIN_SC, VAL_SC
from ..data.coord_transform import nusc_3dbbox_to_2dbbox
from ..visualize.visualize_bbox import draw_box, draw_boxes_on_img
from ..utils import makedir
from ..data.normalize import img_mean_std_BGR, norm_imgs, sklt_local_to_global, norm_bbox, norm_sklt
from ..data.bbox import bbox2d_relation_multi_seq, pad_neighbor
from .dataset_id import DATASET_TO_ID
from ..data.transforms import RandomHorizontalFlip, RandomResizedCrop, crop_local_ctx
from config import dataset_root

import numpy as np
import pickle
import os
import torch
from tqdm import tqdm
from pyquaternion import Quaternion
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import copy
import pdb

NUSC_ROOT = os.path.join(dataset_root, 'nusc')

# img id (unique)
# obj id (unique)
class NuscDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_root=NUSC_ROOT,
                 subset='train',
                 obs_len=4, pred_len=4, overlap_ratio=0.5, 
                 obs_fps=2,
                 recog_act=0,
                 tte=None,
                 offset_traj=False,
                 min_h=72,
                 min_w=36,
                 min_vis_level=3,
                 sensor='CAM_FRONT',
                 small_set=0,
                 augment_mode='random_hflip',
                 resize_mode='even_padded',
                 ctx_size=(224, 224),
                 target_color_order='BGR', 
                 img_norm_mode='torch',
                 modalities=['img', 'sklt', 'ctx', 'traj', 'ego', 'social'],
                 img_format='',
                 sklt_format='coord',
                 ctx_format='ped_graph',
                 traj_format='ltrb',
                 ego_format='accel',
                 social_format='rel_loc',
                 seg_cls=['person', 'vehicle', 'road', 'traffic_light'],
                 max_n_neighbor=10,
                 min_wh=(72,36),
                 ):
        super().__init__()
        self.data_root = data_root
        self.subset = subset
        if self.subset == 'test':
            self.subset = 'val'
        self.dataset_name = 'nuscenes'
        print(f'------------------Init{self.dataset_name}------------------')
        self.fps = 2
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_interval = self.fps // obs_fps - 1
        # sequence length considering interval
        self._obs_len = self.obs_len * (self.seq_interval + 1)
        self._pred_len = self.pred_len * (self.seq_interval + 1)
        self.tte = tte
        self.overlap_ratio = overlap_ratio
        self.offset_traj = offset_traj
        self.recog_act = recog_act
        self.min_h = min_h
        self.min_w = min_w
        self.min_vis_level = min_vis_level
        self.sensor = sensor
        self.small_set = small_set
        self.augment_mode = augment_mode
        self.resize_mode = resize_mode
        self.ctx_size = ctx_size
        self.target_color_order = target_color_order
        self.img_norm_mode = img_norm_mode
        self.modalities = modalities
        self.img_format = img_format
        self.sklt_format = sklt_format
        self.ctx_format = ctx_format
        self.traj_format = traj_format
        self.ego_format = ego_format
        self.social_format = social_format
        self.seg_cls = seg_cls
        self.max_n_neighbor = max_n_neighbor
        # self.ego_motion_key = 'ego_accel' if 'accel' in self.ego_format else 'ego_speed'
        if self.subset == 'train':
            self.sce_names = TRAIN_SC
        elif self.subset == 'val':
            self.sce_names = VAL_SC
        else:
            raise ValueError(self.subset)
        self.img_size = (900, 1600)
        self.transforms = {'random': 0,
                            'balance': 0,
                            'hflip': None,
                            'resized_crop': {'img': None,
                                            'ctx': None,
                                            'sklt': None}}
        self.img_mean, self.img_std = img_mean_std_BGR(self.img_norm_mode)
        self._load_tk_id_dicts()

        self.nusc_root = NUSC_ROOT
        self.extra_root = os.path.join(NUSC_ROOT, 'extra')
        self.nusc = NuScenes(version='v1.0-trainval', 
                             dataroot=NUSC_ROOT, 
                             verbose=True)
        self.imgnm_to_objid_path = os.path.join(self.extra_root, 
                                                self.subset+'_imgnm_to_objid_to_ann.pkl')

        # get vehicle tracks and pedestrian tracks
        self.p_tracks = self.get_obj_tracks(obj_type='ped')
        self.v_tracks = self.get_obj_tracks(obj_type='veh')


        # add the acceleration to the pedestrian tracks
        if 'accel' in self.ego_format:
            self.p_tracks = self._get_accel(self.p_tracks)


        # get cid to img name to obj id dict
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
        self.num_samples = len(self.samples['obs']['sam_id'])

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
        obs_bbox_offset = copy.deepcopy(torch.tensor(self.samples['obs']['bbox_2d_normed'][idx]).float())
        obs_bbox = copy.deepcopy(torch.tensor(self.samples['obs']['bbox_2d'][idx]).float())  # ltrb
        pred_bbox_offset = copy.deepcopy(torch.tensor(self.samples['pred']['bbox_2d_normed'][idx]).float())
        pred_bbox = copy.deepcopy(torch.tensor(self.samples['pred']['bbox_2d'][idx]).float())
        obs_ego = copy.deepcopy(torch.tensor(self.samples['obs']['ego_motion'][idx]).float().unsqueeze(1))
        assert len(obs_ego.shape) == 2
        sce_id_int = torch.tensor(int(self.samples['obs']['sce_id'][idx][0]))
        ins_id_int = torch.tensor(int(float(self.samples['obs']['ins_id'][idx][0])))
        sam_id_int = torch.tensor(self.samples['obs']['sam_id'][idx])
        
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
                  'vid_id_int': sce_id_int,  # int
                  'ped_id_int': ins_id_int,  # int
                  'img_nm_int': sam_id_int,
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
                                copy.deepcopy(np.array(self.samples['obs']['neighbor_bbox_2d'][idx]).transpose(1,0,2)),
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
            for sam_id in self.samples['obs']['sam_id'][idx]:
                img_path = os.path.join(self.extra_root,
                                        'cropped_images',
                                        self.sensor,
                                        self.resize_mode,
                                        '224w_by_224h',
                                        'ped',
                                        str(self.samples['obs']['ins_id'][idx][0]),
                                        str(sam_id)+'.png'
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
            sklts = []
            interm_dir = 'even_padded/48w_by_48h' \
                if self.sklt_format == 'pseudo_heatmap' else 'even_padded/288w_by_384h'
            for sam_id in self.samples['obs']['sam_id'][idx]:
                sklt_path = os.path.join(self.extra_root,
                                        'sk_'+self.sklt_format.replace('0-1', '')+'s',
                                        interm_dir,
                                        str(self.samples['obs']['ins_id'][idx][0]),
                                        str(sam_id)+'.pkl'
                                        )
                with open(sklt_path, 'rb') as f:
                    heatmap = pickle.load(f)
                sklts.append(heatmap)
            sklts = np.stack(sklts, axis=0)  # T, ...
            pred_sklts = []
            interm_dir = 'even_padded/48w_by_48h' \
                if self.sklt_format == 'pseudo_heatmap' else 'even_padded/288w_by_384h'
            for sam_id in self.samples['pred']['sam_id'][idx]:
                sklt_path = os.path.join(self.extra_root,
                                        'sk_'+self.sklt_format.replace('0-1', '')+'s',
                                        interm_dir,
                                        str(self.samples['pred']['ins_id'][idx][0]),
                                        str(sam_id)+'.pkl'
                                        )
                with open(sklt_path, 'rb') as f:
                    heatmap = pickle.load(f)
                pred_sklts.append(heatmap)
            pred_sklts = np.stack(pred_sklts, axis=0)  # T, ...
            if 'coord' in self.sklt_format:
                obs_skeletons = torch.from_numpy(sklts).float().permute(2, 0, 1)[:2]  # shape: (2, T, nj)
                pred_skeletons = torch.from_numpy(pred_sklts).float().permute(2, 0, 1)[:2]
                # yx -> xy
                obs_skeletons = obs_skeletons.flip(0)
                pred_skeletons = pred_skeletons.flip(0)
                # add offset
                obs_skeletons = sklt_local_to_global(obs_skeletons.float(), obs_bbox_ori.float())
                pred_skeletons = sklt_local_to_global(pred_skeletons.float(), pred_bbox_ori.float())
                if '0-1' in self.sklt_format:
                    obs_skeletons = norm_sklt(obs_skeletons, self.dataset_name)
                    pred_skeletons = norm_sklt(pred_skeletons, self.dataset_name)
            elif self.sklt_format == 'pseudo_heatmap':
                # T C H W -> C T H W
                obs_skeletons = torch.from_numpy(sklts).float().permute(1, 0, 2, 3)  # shape: (17, seq_len, 48, 48)
                pred_skeletons = torch.from_numpy(pred_sklts).float().permute(1, 0, 2, 3)
            sample['obs_skeletons'] = obs_skeletons
            sample['pred_skeletons'] = pred_skeletons
        if 'ctx' in self.modalities:
            if self.ctx_format in ('local', 'ori_local', 'mask_ped', 'ori',
                                   'ped_graph', 'ped_graph_seg'):
                if self.ctx_format in ('ped_graph', 'ped_graph_seg'):
                    ctx_format_dir = 'ori_local'
                else:
                    ctx_format_dir = self.ctx_format
                ctx_imgs = []
                for sam_id in self.samples['obs']['sam_id'][idx]:
                    img_path = os.path.join(self.extra_root,
                                        'context',
                                        self.sensor,
                                        ctx_format_dir,
                                        '224w_by_224h',
                                        'ped',
                                        str(self.samples['obs']['ins_id'][idx][0]),
                                        str(sam_id)+'.png'
                                        )
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
                if self.ctx_format in ('ped_graph', 'ped_graph_seg'):
                    all_c_seg = []
                    ins_id = int(float(self.samples['obs']['ins_id'][idx][0]))
                    sam_id = self.samples['obs']['sam_id'][idx][-1]
                    for c in self.seg_cls:
                        seg_path = os.path.join(self.extra_root,
                                                'cropped_seg',
                                                self.sensor,
                                                c,
                                                'ori_local/224w_by_224h/',
                                                'ped',
                                                str(ins_id),
                                                str(sam_id)+'.pkl')
                        with open(seg_path, 'rb') as f:
                            segmap = pickle.load(f)*1  # h w int
                        all_c_seg.append(torch.from_numpy(segmap))
                    all_c_seg = torch.stack(all_c_seg, dim=-1)  # h w n_cls
                    all_c_seg = torch.argmax(all_c_seg, dim=-1, keepdim=True).permute(2, 0, 1)  # 1 h w
                    ctx_imgs = torch.concat([ctx_imgs[:, -1], all_c_seg], dim=0).unsqueeze(1)  # 4 1 h w
                    if self.ctx_format == 'ped_graph_seg':
                        ctx_imgs = ctx_imgs[-1:]
                sample['obs_context'] = ctx_imgs  # shape [3, obs_len, H, W]
            elif self.ctx_format in \
                ('seg_ori_local', 'seg_local'):
                ctx_imgs = []
                for sam_id in self.samples['obs']['sam_id'][idx]:
                    img_path = os.path.join(self.extra_root,
                                        'context',
                                        self.sensor,
                                        'ori_local',
                                        '224w_by_224h',
                                        'ped',
                                        str(self.samples['obs']['ins_id'][idx][0]),
                                        str(sam_id)+'.png'
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
                    for sam_id in self.samples['obs']['sam_id'][idx]:
                        seg_path = os.path.join(
                            self.extra_root,
                            'seg_sam',
                            self.sensor,
                            c,
                            str(sam_id)+'.pkl'
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
    
    def _load_tk_id_dicts(self):
        f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_scene_id_to_token.pkl', 'rb')
        self.scene_id_to_token = pickle.load(f)
        f.close()
        f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_token_to_scene_id.pkl', 'rb')
        self.token_to_scene_id = pickle.load(f)
        f.close()
        f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_sample_id_to_token.pkl', 'rb')
        self.sample_id_to_token = pickle.load(f)
        f.close()
        f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_token_to_sample_id.pkl', 'rb')
        self.token_to_sample_id = pickle.load(f)
        f.close()
        f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_instance_id_to_token.pkl', 'rb')
        self.instance_id_to_token = pickle.load(f)
        f.close()
        f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_token_to_instance_id.pkl', 'rb')
        self.token_to_instance_id = pickle.load(f)
        f.close()
        f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_ins_to_ann_to_id.pkl', 'rb')
        self.instk_to_anntk_to_id = pickle.load(f)
        f.close()
        f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_ins_to_id_to_ann.pkl', 'rb')
        self.instk_to_id_to_anntk = pickle.load(f)
        f.close()
    
    def get_obj_tracks(self, obj_type='ped'):
        '''
        track structure:
        {
            'sce_id': [[int, ...], ...],
            'ins_id': [[int, ...], ...],
            'sam_id': [[int, ...], ...],
            'img_nm': [[str, ...], ...],
            'bbox_3d': [[list, ...], ...],
            'bbox_2d': [[list, ...], ...],
            'ego_speed': [[float, ...], ...]
        }
            
        '''
        assert obj_type in ('ped', 'veh'), obj_type
        obj_tracks = {'sce_id': [],
                    'ins_id': [],
                    'sam_id': [],
                    'img_nm': [],
                    'bbox_3d': [],
                    'bbox_2d': [],
                    'ego_speed': []}
        # discard the last frame
        # min_track_len = self._obs_len + self._pred_len + 1
        min_track_len = 3
        anns_path = '_'.join(['anns', 
                              self.subset, 
                              obj_type, 
                              self.sensor]
                              ) + '.pkl'
        anns_path = os.path.join(self.extra_root, anns_path)
        with open(anns_path, 'rb') as f:
            instk_to_anntk = pickle.load(f)
        print('Getting tracks')
        for instk in tqdm(instk_to_anntk):
            ann_seqs = instk_to_anntk[instk]
            for seq in ann_seqs:
                # skip too short track
                if len(seq) < min_track_len:
                    continue
                processing = False
                # discard the last frame for each track to calc speed
                for i in range(len(seq)-1):
                    ann = self.nusc.get('sample_annotation', seq[i])
                    bbox, corners3d = \
                        nusc_3dbbox_to_2dbbox(self.nusc, ann['token'])
                    vis_l = int(ann['visibility_token'])
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    # end current track
                    if w < self.min_w or \
                        h < self.min_h or \
                            vis_l < self.min_vis_level:
                        # check if length >= min length
                        if processing and \
                            len(obj_tracks['ego_speed'][-1]) < min_track_len:
                            for k in obj_tracks:
                                # delete the recently ended seq
                                obj_tracks[k].pop(-1)
                        processing = False
                        continue
                    next_ann = self.nusc.get('sample_annotation', seq[i+1])
                    sam = self.nusc.get('sample', ann['sample_token'])
                    sce = self.nusc.get('scene', sam['scene_token'])
                    ego_vel = self.calc_ego_velocity(ann['token'], 
                                                     next_ann['token'])
                    sen_data = self.nusc.get('sample_data', 
                                             sam['data'][self.sensor])
                    img_nm = sen_data['filename']
                    sce_id = self.token_to_scene_id[sce['token']]
                    sam_id = self.token_to_sample_id[sam['token']]
                    ins_id = self.token_to_instance_id[instk]
                    if processing:
                        obj_tracks['sce_id'][-1].append(int(sce_id))
                        obj_tracks['sam_id'][-1].append(int(sam_id))
                        obj_tracks['ins_id'][-1].append(int(ins_id))
                        obj_tracks['bbox_2d'][-1].append(bbox)
                        obj_tracks['bbox_3d'][-1].append(corners3d)
                        obj_tracks['img_nm'][-1].append(img_nm)
                        obj_tracks['ego_speed'][-1].append(ego_vel)
                    else:
                        processing = True
                        obj_tracks['sce_id'].append([int(sce_id)])
                        obj_tracks['sam_id'].append([int(sam_id)])
                        obj_tracks['ins_id'].append([int(ins_id)])
                        obj_tracks['bbox_2d'].append([bbox])
                        obj_tracks['bbox_3d'].append([corners3d])
                        obj_tracks['img_nm'].append([img_nm])
                        obj_tracks['ego_speed'].append([ego_vel])
                # check if length >= min length
                if processing and len(obj_tracks['ego_speed'][-1]) < min_track_len:
                    for k in obj_tracks:
                        # delete the recently ended seq
                        obj_tracks[k].pop(-1)
        return obj_tracks    
    
    def _get_accel(self, p_tracks):
        print('Getting ego acceleration')
        new_tracks = {'sce_id': [],
                    'ins_id': [],
                    'sam_id': [],
                    'img_nm': [],
                    'bbox_3d': [],
                    'bbox_2d': [],
                    'ego_speed': [],
                    'ego_accel': [],
                    }
        # calculate acceleration and record the idx of tracks to remove
        idx_to_remove = []
        for i in range(len(p_tracks['ego_speed'])):
            speed_track = p_tracks['ego_speed'][i]
            if len(speed_track) < 2:
                idx_to_remove.append(i)
                continue
            new_tracks['ego_accel'].append([])
            for j in range(len(speed_track)-1):
                accel = (speed_track[j+1] - speed_track[j]) * 2  # 2 fps
                new_tracks['ego_accel'][-1].append(accel)
        # add the rest keys
        for k in p_tracks:
            for i in range(len(p_tracks[k])):
                if i in idx_to_remove:
                    continue
                cur_track = p_tracks[k][i]
                new_tracks[k].append(cur_track[1:])

        # check if the lengths of all keys confront
        for k in new_tracks:
            assert len(new_tracks[k]) == len(new_tracks['ego_accel']), \
                (len(new_tracks[k]), len(new_tracks['ego_accel']))
            for i in range(len(new_tracks[k])):
                assert len(new_tracks[k][i]) == len(new_tracks['ego_accel'][i]), \
                    (len(new_tracks[k][i]), len(new_tracks['ego_accel'][i]))
        return new_tracks

    def calc_ego_velocity(self, cur_anntk, next_anntk):
        cur_ann = self.nusc.get('sample_annotation', cur_anntk)
        next_ann = self.nusc.get('sample_annotation', next_anntk)
        cur_sam = self.nusc.get('sample', cur_ann['sample_token'])
        next_sam = self.nusc.get('sample', next_ann['sample_token'])
        cur_sen_data = self.nusc.get('sample_data', cur_sam['data'][self.sensor])
        next_sen_data = self.nusc.get('sample_data', next_sam['data'][self.sensor])
        cur_pose = self.nusc.get('ego_pose', cur_sen_data['ego_pose_token'])
        next_pose = self.nusc.get('ego_pose', next_sen_data['ego_pose_token'])

        cur_loc = np.array(cur_pose['translation'])
        next_loc = np.array(next_pose['translation'])

        cur_time = 1e-6 * cur_pose['timestamp']
        next_time = 1e-6 * next_pose['timestamp']

        ego_vel = (next_loc - cur_loc) / (next_time - cur_time)

        return np.sqrt(ego_vel[0]**2 + ego_vel[1]**2) 

    def get_imgnm_to_objid(self, p_tracks, v_tracks, save_path):
        # 
        imgnm_to_oid_to_info = {}
        # pedestrian
        tracks = p_tracks
        n_tracks = len(tracks['bbox_2d'])
        print('Saving imgnm to objid to obj info of pedestrians in nuScenes')
        for i in tqdm(range(n_tracks)):
            ins_id = str(tracks['ins_id'][i][0])
            sam_ids = tracks['sam_id'][i]
            for j in range(len(sam_ids)):
                sam_id = str(sam_ids[j])
                # initialize img dict
                if sam_id not in imgnm_to_oid_to_info:
                    imgnm_to_oid_to_info[sam_id] = {}
                    imgnm_to_oid_to_info[sam_id]['ped'] = {}
                    imgnm_to_oid_to_info[sam_id]['veh'] = {}
                # initialize obj dict
                imgnm_to_oid_to_info[sam_id]['ped'][ins_id] = {}
                imgnm_to_oid_to_info[sam_id]['ped'][ins_id]['bbox_2d'] = \
                    tracks['bbox_2d'][i][j]
                imgnm_to_oid_to_info[sam_id]['ped'][ins_id]['bbox_3d'] = \
                    tracks['bbox_3d'][i][j]
        # vehicle
        tracks = v_tracks
        n_tracks = len(tracks['bbox_2d'])
        print('Saving imgnm to objid to obj info of vehicles in nuScenes')
        for i in tqdm(range(n_tracks)):
            ins_id = str(tracks['ins_id'][i][0])
            sam_ids = tracks['sam_id'][i]
            for j in range(len(sam_ids)):
                sam_id = str(sam_ids[j])
                # initialize img dict
                if sam_id not in imgnm_to_oid_to_info:
                    imgnm_to_oid_to_info[sam_id] = {}
                    imgnm_to_oid_to_info[sam_id]['ped'] = {}
                    imgnm_to_oid_to_info[sam_id]['veh'] = {}
                # initialize obj dict
                imgnm_to_oid_to_info[sam_id]['veh'][ins_id] = {}
                imgnm_to_oid_to_info[sam_id]['veh'][ins_id]['bbox_2d'] = \
                    tracks['bbox_2d'][i][j]
                imgnm_to_oid_to_info[sam_id]['veh'][ins_id]['bbox_3d'] = \
                    tracks['bbox_3d'][i][j]
        with open(save_path, 'wb') as f:
            pickle.dump(imgnm_to_oid_to_info, f)
        return imgnm_to_oid_to_info

    def get_neighbor_relation(self,
                      samples,
                      save_path,
                      padding_val=0):
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                neighbor_seq = pickle.load(f)
        else:
            n_sample = len(samples['obs']['bbox_2d'])
            relations = []
            neighbor_bbox = []
            neighbor_oid = []
            neighbor_cls = []
            print('Getting neighbor sequences')
            for i in tqdm(range(n_sample)):
                target_oid = str(samples['obs']['ins_id'][i][0])  # str
                target_bbox_seq = np.array(samples['obs']['bbox_2d'][i])  # T, 4
                obslen = len(target_bbox_seq)
                bbox_seq_dict = {'ped':{},
                                'veh':{}}
                for j in range(obslen):
                    imgnm = str(samples['obs']['sam_id'][i][j])  # str
                    cur_ped_ids = set(self.imgnm_to_objid[imgnm]['ped'].keys())
                    try:
                        cur_ped_ids.remove(target_oid)
                    except:
                        print(cur_ped_ids, target_oid, imgnm, self.dataset_name)
                        import pdb; pdb.set_trace()
                        raise ValueError()
                    cur_veh_ids = set(self.imgnm_to_objid[imgnm]['veh'].keys())
                    # ped neighbor for cur sample
                    # existing neighbor
                    for oid in set(bbox_seq_dict['ped'].keys())&cur_ped_ids:
                        bbox = np.array(self.imgnm_to_objid[imgnm]['ped'][oid]['bbox_2d'])
                        bbox_seq_dict['ped'][oid].append(bbox)
                    # first appearing neighbor
                    for oid in cur_ped_ids-set(bbox_seq_dict['ped'].keys()):
                        bbox = np.array(self.imgnm_to_objid[imgnm]['ped'][oid]['bbox_2d'])
                        bbox_seq_dict['ped'][oid] = [np.ones([4])*padding_val]*j + [bbox]  # T, 4
                    # disappeared neighbor
                    for oid in set(bbox_seq_dict['ped'].keys())-cur_ped_ids:
                        bbox_seq_dict['ped'][oid].append(np.ones([4])*padding_val)
                    
                    # veh neighbor for cur sample
                    # existing neighbor
                    for oid in set(bbox_seq_dict['veh'].keys())&cur_veh_ids:
                        bbox = np.array(self.imgnm_to_objid[imgnm]['veh'][oid]['bbox_2d'])
                        bbox_seq_dict['veh'][oid].append(bbox)
                    # first appearing neighbor
                    for oid in cur_veh_ids-set(bbox_seq_dict['veh'].keys()):
                        bbox = np.array(self.imgnm_to_objid[imgnm]['veh'][oid]['bbox_2d'])
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
                try:
                    cur_relations = bbox2d_relation_multi_seq(target_bbox_seq,
                                                            cur_neighbor_bbox,
                                                            rela_func='log_bbox_reg')  # K T 4
                except:
                    print(cur_neighbor_bbox.shape)
                    import pdb; pdb.set_trace()
                    raise ValueError()
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
            neighbor_seq['neighbor_bbox_2d'] = neighbor_bbox
            neighbor_seq['neighbor_oid'] = neighbor_oid
            neighbor_seq['neighbor_cls'] = neighbor_cls
            with open(save_path, 'wb') as f:
                pickle.dump(neighbor_seq, f)
        # add neighbor info to samples
        for k in neighbor_seq:
            samples['obs'][k] = neighbor_seq[k]
        return samples

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

        # normalize tracks by subtracting bbox/center of the 1st frame
        print('---------------Normalize traj---------------')
        bboxes_2d_norm = copy.deepcopy(samples['bbox_2d'])
        bboxes_3d_norm = copy.deepcopy(samples['bbox_3d'])
        if self.offset_traj:
            for i in range(len(bboxes_2d_norm)):
                bboxes_2d_norm[i] = np.subtract(bboxes_2d_norm[i][:], 
                                                bboxes_2d_norm[i][0]).tolist()
                bboxes_3d_norm[i] = np.subtract(bboxes_3d_norm[i][:], 
                                                bboxes_3d_norm[i][0]).tolist()
        samples['bbox_2d_normed'] = bboxes_2d_norm
        samples['bbox_3d_normed'] = bboxes_3d_norm

        # choose ego motion quantity
        samples['ego_motion'] = copy.deepcopy(samples['ego_accel']) \
            if 'accel' in self.ego_format else copy.deepcopy(samples['ego_speed'])
        
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
    


def save_scene_token_dict():
    '''
    scene id to token/token to id dict
    '''
    scene_id_to_token = {}
    token_to_scene_id = {}
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)
    for i in range(len(nusc.scene)):
        cur_t = nusc.scene[i]['token']
        scene_id_to_token[str(i)] = cur_t
        token_to_scene_id[cur_t] = str(i)
        print(f'scene {i} done')
    save_path = os.path.join(dataset_root, 'nusc/extra')
    with open(os.path.join(save_path, 'trainval_scene_id_to_token.pkl'), 'wb') as f:
        pickle.dump(scene_id_to_token, f)
    with open(os.path.join(save_path, 'trainval_token_to_scene_id.pkl'), 'wb') as f:
        pickle.dump(token_to_scene_id, f)

def save_sample_token_dict():
    '''
    sample id to token/token to id dict
    '''
    sample_id_to_token = {}
    token_to_sample_id = {}
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)
    for i in range(len(nusc.sample)):
        cur_t = nusc.sample[i]['token']
        sample_id_to_token[str(i)] = cur_t
        token_to_sample_id[cur_t] = str(i)
        print(f'sample {i} done')
    save_path = os.path.join(dataset_root, 'nusc/extra')
    with open(os.path.join(save_path, 'trainval_sample_id_to_token.pkl'), 'wb') as f:
        pickle.dump(sample_id_to_token, f)
    with open(os.path.join(save_path, 'trainval_token_to_sample_id.pkl'), 'wb') as f:
        pickle.dump(token_to_sample_id, f)

def save_instance_token_dict():
    '''
    instance id to token/token to id dict
    '''
    instance_id_to_token = {}
    token_to_instance_id = {}
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)
    for i in range(len(nusc.instance)):
        cur_t = nusc.instance[i]['token']
        instance_id_to_token[str(i)] = cur_t
        token_to_instance_id[cur_t] = str(i)
        print(f'instance {i} done')
    save_path = os.path.join(dataset_root, 'nusc/extra')
    with open(os.path.join(save_path, 'trainval_instance_id_to_token.pkl'), 'wb') as f:
        pickle.dump(instance_id_to_token, f)
    with open(os.path.join(save_path, 'trainval_token_to_instance_id.pkl'), 'wb') as f:
        pickle.dump(token_to_instance_id, f)

def _save_ins_to_ann_to_dict():
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)
    instk_to_anntk_to_id = {}
    instk_to_id_to_anntk = {}
    for ins in tqdm(nusc.instance):
        instk_to_anntk_to_id[ins['token']] = {}
        instk_to_id_to_anntk[ins['token']] = {}
        cur_ann_tk = ins['first_annotation_token']
        i = 0
        while cur_ann_tk != '':
            instk_to_anntk_to_id[ins['token']][cur_ann_tk] = str(i)
            instk_to_id_to_anntk[ins['token']][str(i)] = cur_ann_tk
            cur_ann_tk = nusc.get('sample_annotation', cur_ann_tk)['next']
    save_path = '/home/y_feng/workspace6/datasets/nusc/extra/token_id'
    with open(os.path.join(save_path, 'trainval_ins_to_ann_to_id.pkl'), 'wb') as f:
        pickle.dump(instk_to_anntk_to_id, f)
    with open(os.path.join(save_path, 'trainval_ins_to_id_to_ann.pkl'), 'wb') as f:
        pickle.dump(instk_to_id_to_anntk, f)

def save_ins_tokens(subset='train', cate='ped'):
    '''
    save instance tokens of specific subset and category
    '''
    _cate = 'human' if cate == 'ped' else 'vehicle'
    tokens_to_save = []
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)
    for ins in nusc.instance:
        category = nusc.get('category', ins['category_token'])
        first_ann = nusc.get('sample_annotation', ins['first_annotation_token'])
        first_sam = nusc.get('sample', first_ann['sample_token'])
        cur_sce = nusc.get('scene', first_sam['scene_token'])
        if subset == 'train':
            scenes_cur_subset = TRAIN_SC
        elif subset == 'val':
            scenes_cur_subset = VAL_SC
        if cur_sce['name'] not in scenes_cur_subset or category['name'].split('.')[0] != _cate:
            print('ins '+ins['token']+' not saved')
            continue
        tokens_to_save.append(ins['token'])
        print(f'ins {tokens_to_save[-1]} saved')
    if _cate == 'human':
        fnm = subset+'_'+'ped_ins_token.pkl'
    elif _cate == 'vehicle':
        fnm = subset+'_'+'veh_ins_token.pkl'
    save_path = os.path.join(NUSC_ROOT, 'extra', fnm)
    with open(save_path, 'wb') as f:
        pickle.dump(tokens_to_save, f)

def save_anns_in_sensor(ins_tokens_path=os.path.join(dataset_root, 
                                                     'nusc/extra/train_ped_ins_token.pkl'), 
                        sensor='CAM_FRONT'):
    '''
    save annotation sequences observed by specific sensor
    '''
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)
    ins_tokens_nm = ins_tokens_path.split('/')[-1]
    subset, cate, _, _ = ins_tokens_nm.split('_')
    with open(ins_tokens_path, 'rb') as f:
        ins_tokens = pickle.load(f)
    ins_tk_to_ann_tks = {}
    print(f'Processing {ins_tokens_nm}')
    for ins_tk in tqdm(ins_tokens):
        cur_ins = nusc.get('instance', ins_tk)
        ann_tk_seqs = []
        cur_ann_tk = cur_ins['first_annotation_token']
        processing = False
        while cur_ann_tk != '':
            cur_ann = nusc.get('sample_annotation', cur_ann_tk)
            is_observed = is_observed_by_sensor(nusc, cur_ann_tk, sensor)
            if not is_observed:
                processing = False
            elif processing:
                ann_tk_seqs[-1].append(cur_ann_tk)
            else:
                ann_tk_seqs.append([cur_ann_tk])
                processing = True
            cur_ann_tk = cur_ann['next']
        if len(ann_tk_seqs) > 1:
            print(f'ins {ins_tk} has {len(ann_tk_seqs)} seqs')
        ins_tk_to_ann_tks[ins_tk] = ann_tk_seqs
    file_nm = 'anns_' + subset + '_' + cate + '_' + sensor + '.pkl'
    save_path = ins_tokens_path.replace(ins_tokens_nm, file_nm)
    with open(save_path, 'wb') as f:
        pickle.dump(ins_tk_to_ann_tks, f)
    
def is_observed_by_sensor(nusc, ann_tk, sensor):
    ann = nusc.get('sample_annotation', ann_tk)
    sam = nusc.get('sample', ann['sample_token'])
    _, bbs, _ = nusc.get_sample_data(sam['data'][sensor])
    for bb in bbs:
        if bb.token == ann['token']:
            return True
    return False

def check_3d_to_2d():
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)
    anns_path = '/home/y_feng/workspace6/datasets/nusc/extra/anns_train_ped_CAM_FRONT.pkl'
    res_path = './module_test'
    makedir(res_path)
    with open(anns_path, 'rb') as f:
        ins_to_anns = pickle.load(f)
    instk = list(ins_to_anns.keys())[1]
    for i in range(len(ins_to_anns[instk][0])):
        anntk = ins_to_anns[instk][0][i]
        ann = nusc.get('sample_annotation', anntk)
        attr = nusc.get('attribute', ann['attribute_tokens'][0])
        print(f'\nann {ann}')
        print(f'attr {attr}')
        
        is_obs = is_observed_by_sensor(nusc, anntk, 'CAM_FRONT')
        print(f'is obs {is_obs}')
        sam = nusc.get('sample', ann['sample_token'])
        cam_front_data = nusc.get('sample_data', sam['data']['CAM_FRONT'])
        cali_sensor = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
        camera_intrinsic = np.array(cali_sensor['camera_intrinsic'])
        ego_pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
        img_path = os.path.join(NUSC_ROOT, cam_front_data['filename'])
        img = cv2.imread(img_path)

        # Get the annotation box
        box = nusc.get_box(ann['token'])
        
        # Convert coords from map to veh to cam
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)
        box.translate(-np.array(cali_sensor['translation']))
        box.rotate(Quaternion(cali_sensor['rotation']).inverse)

        corners_3d = box.corners()  # 3, 8 corners in camera coord

        # Project to image plane
        view = np.eye(4)
        view[:3, :3] = np.array(camera_intrinsic)
        in_front = corners_3d[2, :] > 0.1  # ensure z > 0.1
        assert all(in_front), corners_3d
        corners = view_points(corners_3d, view, normalize=True)[:2, :]  # 2, 8
        # Get the 2D bounding box coordinates
        x1 = min(corners[0])
        x2 = max(corners[0])
        y1 = min(corners[1])
        y2 = max(corners[1])
        box2d = [x1, y1, x2, y2]  # ltrb
        print(box2d)
        img = draw_box(img, box2d)
        cv2.imwrite(os.path.join(res_path, '2dbbox'+str(i)+'.png'), img)

# def save_cropped_imgs(obj_type='ped', sensor='CAM_FRONT', modality='img', resize_mode='even_padded', target_size=(224, 224)):
#     print(f'{obj_type}, {sensor}, {modality}, {resize_mode}')
#     nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)

#     f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_token_to_sample_id.pkl', 'rb')
#     token_to_sample_id = pickle.load(f)
#     f.close()

#     f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_token_to_instance_id.pkl', 'rb')
#     token_to_instance_id = pickle.load(f)
#     f.close()

#     f = open(os.path.join('/home/y_feng/workspace6/datasets/nusc/extra', '_'.join(['anns_train', obj_type, sensor])+'.pkl'), 'rb')
#     instk_to_anntks = pickle.load(f)
#     f.close()
#     f = open(os.path.join('/home/y_feng/workspace6/datasets/nusc/extra', '_'.join(['anns_val', obj_type, sensor])+'.pkl'), 'rb')
#     val_instk_to_anntks = pickle.load(f)
#     f.close()
#     instk_to_anntks.update(val_instk_to_anntks)

#     if modality == 'img':
#         root = '/home/y_feng/workspace6/datasets/nusc/extra/cropped_images'
#         root = os.path.join(root, sensor, resize_mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h', obj_type)
#         makedir(root)
#     elif modality == 'ctx':
#         root = '/home/y_feng/workspace6/datasets/nusc/extra/context'
#         root = os.path.join(root, sensor, resize_mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h', obj_type)
#         makedir(root)

#     for instk in tqdm(instk_to_anntks):
#         insid = token_to_instance_id[instk]
#         ins_path = os.path.join(root, insid)
#         makedir(ins_path)
#         anntk_seqs = instk_to_anntks[instk]
#         seq_all = []
#         for seq in anntk_seqs:
#             seq_all += seq
#         for anntk in seq_all:
#             bbox = nusc_3dbbox_to_2dbbox(nusc, anntk)[0]  # ltrb
#             l, t, r, b = bbox
#             if r-l<2 or b-t<2:
#                 continue
#             ann = nusc.get('sample_annotation', anntk)
#             samtk = nusc.get('sample', ann['sample_token'])['token']
#             sam = nusc.get('sample', samtk)
#             samid = token_to_sample_id[samtk]
#             sen_data = nusc.get('sample_data', sam['data'][sensor])
#             img_path = os.path.join(NUSC_ROOT, sen_data['filename'])
#             img = cv2.imread(img_path)
#             if modality == 'img':
#                 cropped_img = crop_img(img, bbox, resize_mode, target_size)
#             elif modality == 'ctx':
#                 cropped_img = crop_ctx(img, bbox, resize_mode, target_size)
#             if cropped_img is None:
#                 continue
#             save_path = os.path.join(ins_path, samid+'.png')
#             cv2.imwrite(save_path, cropped_img)


if __name__ == '__main__':
    # save_cropped_imgs(obj_type='human', sensor='CAM_FRONT', modality='img', resize_mode='even_padded', target_size=(224, 224))
    # save_cropped_imgs(obj_type='human', sensor='CAM_FRONT', modality='img', resize_mode='even_padded', target_size=(288, 384))
    # save_cropped_imgs(obj_type='human', sensor='CAM_FRONT', modality='ctx', resize_mode='local', target_size=(224, 224))
    # save_cropped_imgs(obj_type='human', sensor='CAM_FRONT', modality='ctx', resize_mode='ori_local', target_size=(224, 224))
    # save_cropped_imgs(obj_type='vehicle', sensor='CAM_FRONT', modality='img', resize_mode='even_padded', target_size=(224, 224))
    # ins_tk_paths = ['/home/y_feng/workspace6/datasets/nusc/extra/train_human_ins_token.pkl',
    #                 '/home/y_feng/workspace6/datasets/nusc/extra/train_vehicle_ins_token.pkl',
    #                 '/home/y_feng/workspace6/datasets/nusc/extra/val_human_ins_token.pkl',
    #                 '/home/y_feng/workspace6/datasets/nusc/extra/val_vehicle_ins_token.pkl']
    # for path in ins_tk_paths:
    #     save_anns_in_sensor(path, sensor='CAM_FRONT')
    # save_ins_tokens(subset='train', cate='human')
    # save_ins_tokens(subset='val', cate='human')
    # save_ins_tokens(subset='train', cate='vehicle')
    # save_ins_tokens(subset='val', cate='vehicle')
    # save_scene_token_dict()
    # save_sample_token_dict()
    # save_instance_token_dict()
    # sensors = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'LIDAR_TOP', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)

    ins = nusc.instance[1]
    print(f'\ninstance {ins}')
    ann = nusc.get('sample_annotation', ins['first_annotation_token'])
    print(f'\nannotation {ann}')
    attr = nusc.get('attribute', ann['attribute_tokens'][0])
    print(f'\nattr {attr}')
    sam = nusc.get('sample', ann['sample_token'])
    print(f'\nsample {sam}')
    sce = nusc.get('scene', sam['scene_token'])
    print(f'\nscene {sce}')
    cam_front_data = nusc.get('sample_data', 
                              sam['data']['CAM_FRONT'])
    cam_front_cali = nusc.get('calibrated_sensor', 
                              cam_front_data['calibrated_sensor_token'])
    cam_back_data = nusc.get('sample_data', 
                             sam['data']['CAM_BACK'])
    print(f'\ncam front data {cam_front_data}')
    print(f'\ncam front cali: {cam_front_cali}')
    pdb.set_trace()
    # ego_pose_front = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
    # print(f'\nego pose {ego_pose_front}')
    # sam_next = nusc.get('sample', sam['next'])
    # cam_front_data_next = nusc.get('sample_data', sam_next['data']['CAM_FRONT'])


    # print(len(cam_front_data_[1]))
    # print(type(cam_front_data_[1][0]))
    # print(cam_front_data_[1])
    # for bb in cam_front_data_[1]:
    #     print(bb.token)
    
    # for i in range(len(nusc.instance)):
    #     ins = nusc.instance[i]
    #     cur_ann_token = ins['first_annotation_token']
    #     # ann_tokens = nusc.field2token('sample_annotation', 'instance_token', ins['token'])
    #     j = 0
    #     while cur_ann_token != '':
    #         # ann = nusc.get('sample_annotation', ann_tokens[j])
    #         ann = nusc.get('sample_annotation', cur_ann_token)
    #         sam = nusc.get('sample', ann['sample_token'])
    #         cur_sensors = []
    #         for sensor in sensors:
    #             _, bbs, _ = nusc.get_sample_data(sam['data'][sensor])
    #             for bb in bbs:
    #                 if bb.token == ann['token']:
    #                     cur_sensors.append(sensor)
    #                     break
    #         print(f'instance {i} ann {j} is observed by {cur_sensors}')
    #         cur_ann_token = ann['next']
    #         j += 1
    # res_path = '/home/y_feng/workspace6/work/ProtoPNet/ProtoPNet/module_test/check_divided_track'
    # makedir(res_path)
    # nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)
    # anns_path = '/home/y_feng/workspace6/datasets/nusc/extra/anns_train_human_CAM_FRONT.pkl'
    # n_ann = 0
    # with open(anns_path, 'rb') as f:
    #     instk_to_anntk = pickle.load(f)
    # for instk in instk_to_anntk:
    #     anntk_seqs = instk_to_anntk[instk]
    #     for seq in anntk_seqs:
    #         n_ann += len(seq)
    # print(n_ann)
    #     if len(anntk_seqs) > 1:
    #         print([len(anns) for anns in anntk_seqs])
    #         cur_ins_path = os.path.join(res_path, instk)
    #         makedir(cur_ins_path)
    #         for i in range(len(anntk_seqs)):
    #             for j in range(len(anntk_seqs[i])):
    #                 img_nm = f'{i}_{j}.png'
    #                 anntk = anntk_seqs[i][j]
    #                 box = nusc_3dbbox_to_2dbbox(nusc, anntk)
    #                 ann = nusc.get('sample_annotation', anntk)
    #                 sam = nusc.get('sample', ann['sample_token'])
    #                 cam_front_data = nusc.get('sample_data', sam['data']['CAM_FRONT'])
    #                 img_path = os.path.join(NUSC_ROOT, cam_front_data['filename'])
    #                 img = cv2.imread(img_path)
    #                 img = draw_box(img, box)
    #                 save_path = os.path.join(cur_ins_path, img_nm)
    #                 cv2.imwrite(save_path, img)