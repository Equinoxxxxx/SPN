import cv2
import numpy as np
import os
import pickle
import json
import pdb
from tqdm import tqdm
import torch
import torchvision
import argparse
from ..utils import makedir
from config import dataset_root
from config import cktp_root

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from nuscenes.nuscenes import NuScenes
from ..datasets.TITAN import TITAN_dataset
from ..datasets.pie_data import PIE
from ..datasets.jaad_data import JAAD
from .crop_images import crop_img, crop_ctx
from .coord_transform import nusc_3dbbox_to_2dbbox


def segment_dataset(datasets='bdd100k',
                    prompt=['person', 'vehicle', 'road', 'traffic light'],
                    nusc_sensor='CAM_FRONT'):

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = "./Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(cktp_root, "groundingdino_swint_ogc.pth")

    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = os.path.join(cktp_root, "sam_vit_h_4b8939.pth")

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(DEVICE)
    sam_predictor = SamPredictor(sam)


    # Predict classes and hyper-param for GroundingDINO
    CLASSES = [
               'person', 
               'vehicle', 
               'road', 
            #    'cross walks', 
               'traffic light',
            #    'sidewalks'
               ]
    
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.8

    # dataset setting
    DATASET_2_SRCPATH = {
        'TITAN': os.path.join(dataset_root, 'TITAN/honda_titan_dataset/dataset/images_anonymized'),
        'PIE': os.path.join(dataset_root, 'PIE_dataset/images'),
        'JAAD': os.path.join(dataset_root, 'JAAD/images'),
        'nuscenes': os.path.join(dataset_root, 'nusc'),
        'bdd100k': os.path.join(dataset_root, 'BDD100k/bdd100k/images/track')
    }
    DATASET_2_TGTPATH = {
        'TITAN': os.path.join(dataset_root, 'TITAN/TITAN_extra/seg_sam'),
        'PIE': os.path.join(dataset_root, 'PIE_dataset/seg_sam'),
        'JAAD': os.path.join(dataset_root, 'JAAD/seg_sam'),
        'nuscenes': os.path.join(dataset_root, 'nusc/extra/seg_sam/', nusc_sensor),
        'bdd100k': os.path.join(dataset_root, 'BDD100k/bdd100k/extra/seg_sam')
    }

    # assert prompt in CLASSES


    def seg_img(sam_predictor: SamPredictor, 
                image: np.ndarray, 
                ):
        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=BOX_THRESHOLD
        )

        # NMS post process
        # print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # print(f"After NMS: {len(detections.xyxy)} boxes")

        # Prompting SAM with detected boxes
        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, logits = sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )  # 每个cls预测多张mask，取score最高的那一张输出
                index = np.argmax(scores)
                result_masks.append(masks[index])
                # pdb.set_trace()
            # print(len(result_masks), xyxy.shape, image.shape)
            if len(result_masks) == 0:
                result_masks = np.zeros(shape=(len(prompt), image.shape[0], image.shape[1]))
            return np.array(result_masks)


        # convert detections to masks
        detec_masks = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )  # bool ndarray: n_detect,H,W
        
        # converge masks
        res = np.full((len(CLASSES),)+image.shape[:2], False)  # n_cls H W
        for i in range(len(detections.class_id)):
            try:
                cls = detections.class_id[i]
                res[cls] = res[cls]|detec_masks[i]
            except IndexError:
                pdb.set_trace()

        return res
        

    if 'TITAN' in datasets:
        print('TITAN')
        srcpath = DATASET_2_SRCPATH['TITAN']
        tgtpath = DATASET_2_TGTPATH['TITAN']
        for clip in tqdm(os.listdir(srcpath), desc='video loop'):
            v_path = os.path.join(srcpath, clip, 'images')
            v_id = clip.replace('clip_', '')
            tgt_v_dirs = []
            for cls in prompt:
                tgt_v_dir=os.path.join(tgtpath, cls.replace(' ', '_'), v_id)
                makedir(tgt_v_dir)
                tgt_v_dirs.append(tgt_v_dir)

            for img_nm in tqdm(os.listdir(v_path), desc='img loop'):
                src_img = cv2.imread(os.path.join(v_path, img_nm))
                img_id = img_nm.replace('.png', '')

                masks = seg_img(sam_predictor=sam_predictor,
                                image=src_img)  # n_cls H W
                for c in range(len(prompt)):
                    mask = masks[c]
                    tgt_f_path = os.path.join(tgt_v_dirs[c],
                                              img_id+'.pkl')
                    with open(tgt_f_path, 'wb') as f:
                        pickle.dump(mask, f)  # H W

    
    if 'PIE' in datasets:
        print('PIE')
        srcpath = DATASET_2_SRCPATH['PIE']
        tgtpath = DATASET_2_TGTPATH['PIE']
        for s in tqdm(os.listdir(srcpath), desc='set loop'):
            set_path = os.path.join(srcpath, s)
            set_id = s.replace('set', '')
            set_id = str(int(set_id))
            tgt_s_dirs = []
            for cls in prompt:
                tgt_s_dir = os.path.join(tgtpath, cls.replace(' ', '_'), set_id)
                tgt_s_dirs.append(tgt_s_dir)
            for clip in tqdm(os.listdir(set_path), desc='video loop'):
                v_path = os.path.join(set_path, clip)
                v_id = clip.replace('video_', '')
                v_id = str(int(v_id))
                tgt_v_dirs = []
                for tgt_s_dir in tgt_s_dirs:
                    tgt_v_dir = os.path.join(tgt_s_dir, v_id)
                    makedir(tgt_v_dir)
                    tgt_v_dirs.append(tgt_v_dir)

                for img_nm in tqdm(os.listdir(v_path), desc='img loop'):
                    src_img = cv2.imread(os.path.join(v_path, img_nm))
                    img_id = img_nm.replace('.png', '')

                    masks = seg_img(sam_predictor=sam_predictor,
                                    image=src_img)  # 1 H W
                    for c in range(len(prompt)):
                        mask = masks[c]
                        tgt_f_path = os.path.join(tgt_v_dirs[c], 
                                                  img_id+'.pkl')
                        with open(tgt_f_path, 'wb') as f:
                            pickle.dump(mask, f)  # H W
        
    if 'JAAD' in datasets:
        print('JAAD')
        srcpath = DATASET_2_SRCPATH['JAAD']
        tgtpath = DATASET_2_TGTPATH['JAAD']
        for clip in tqdm(os.listdir(srcpath), desc='video loop'):
            v_path = os.path.join(srcpath, clip)
            v_id = clip.replace('video_', '')
            v_id = str(int(v_id))
            tgt_v_dirs = []
            for cls in prompt:
                tgt_v_dir = os.path.join(tgtpath, cls.replace(' ', '_'), v_id)
                makedir(tgt_v_dir)
                tgt_v_dirs.append(tgt_v_dir)

            for img_nm in tqdm(os.listdir(v_path), desc='img loop'):
                src_img = cv2.imread(os.path.join(v_path, img_nm))
                img_id = img_nm.replace('.png', '')

                masks = seg_img(sam_predictor=sam_predictor,
                                image=src_img)  # 1 H W
                for c in range(len(prompt)):
                    mask = masks[c]
                    tgt_f_path = os.path.join(tgt_v_dirs[c],
                                              img_id+'.pkl')
                    with open(tgt_f_path, 'wb') as f:
                        pickle.dump(mask, f)  # H W

    if 'nuscenes' in datasets:
        print('nuscenes')
        srcpath = DATASET_2_SRCPATH['nuscenes']
        tgtpath = DATASET_2_TGTPATH['nuscenes']
        nusc = NuScenes(version='v1.0-trainval', dataroot=srcpath, verbose=True)
        instk_to_anntk_paths = [os.path.join(srcpath, 'extra/anns_train_ped_CAM_FRONT.pkl'),
                                os.path.join(srcpath, 'extra/anns_train_veh_CAM_FRONT.pkl'),
                                os.path.join(srcpath, 'extra/anns_val_ped_CAM_FRONT.pkl'),
                                os.path.join(srcpath, 'extra/anns_val_veh_CAM_FRONT.pkl'),
                                ]
        token_to_samid_path = os.path.join(srcpath, 'extra/token_id/trainval_token_to_sample_id.pkl')
        samid_to_token_path = os.path.join(srcpath, 'extra/token_id/trainval_sample_id_to_token.pkl')
        with open(token_to_samid_path, 'rb') as f:
            token_to_samid = pickle.load(f)
        with open(samid_to_token_path, 'rb') as f:
            samid_to_token = pickle.load(f)
        samids = []
        for path in instk_to_anntk_paths:
            with open(path, 'rb') as f:
                instk_to_anntks = pickle.load(f)
            for instk in instk_to_anntks:
                anntk_seqs = instk_to_anntks[instk]
                for seq in anntk_seqs:
                    for anntk in seq:
                        ann = nusc.get('sample_annotation', anntk)
                        samid = token_to_samid[ann['sample_token']]
                        samids.append(samid)
        samids = set(samids)
        for samid in tqdm(samids):
            samtk = samid_to_token[samid]
            sam = nusc.get('sample', samtk)
            sen_data = nusc.get('sample_data', sam['data'][nusc_sensor])
            img_path = os.path.join(srcpath, sen_data['filename'])
            src_img = cv2.imread(img_path)
            masks = seg_img(sam_predictor=sam_predictor,
                                image=src_img)  # n_cls H W
            for c in range(len(prompt)):
                cls = prompt[c].replace(' ', '_')
                mask = masks[c]
                makedir(os.path.join(tgtpath,
                                     cls))
                tgt_f_path = os.path.join(tgtpath, 
                                          cls, 
                                          samid+'.pkl')
                with open(tgt_f_path, 'wb') as f:
                    pickle.dump(mask, f)
    
    if 'bdd100k' in datasets:
        print('bdd100k')
        srcpath = DATASET_2_SRCPATH['bdd100k']
        tgtpath = DATASET_2_TGTPATH['bdd100k']
        vid_nm2id_path = os.path.join(dataset_root, 'BDD100k', 'bdd100k', 'extra', 'vid_nm2id.pkl')
        with open(vid_nm2id_path, 'rb') as f:
            vid_nm2id = pickle.load(f)
        for _subset in os.listdir(srcpath):
            print(_subset)
            for vid_nm in tqdm(os.listdir(os.path.join(srcpath, 
                                                       _subset))[209:],
                                ascii=True,desc="video loop"):
                vid_dir = os.path.join(srcpath, 
                                       _subset, 
                                       vid_nm)
                vid_id = vid_nm2id[vid_nm]
                for img_nm in tqdm(os.listdir(vid_dir),
                                   ascii=True,desc="img loop"):
                    img_path = os.path.join(vid_dir, img_nm)
                    img_id = img_nm.split('-')[-1].replace('.jpg', '')
                    img_id_int = int(img_id)
                    src_img = cv2.imread(img_path)
                    masks = seg_img(sam_predictor=sam_predictor,
                                        image=src_img)  # n_cls H W
                    for c in range(len(prompt)):
                        cls = prompt[c].replace(' ', '_')
                        mask = masks[c]
                        makedir(os.path.join(tgtpath,
                                             cls,
                                             str(vid_id)))
                        tgt_f_path = os.path.join(tgtpath,
                                                    cls,
                                                    str(vid_id), 
                                                    str(img_id_int)+'.pkl')
                        # print(src_img.shape, masks.shape, img_path)
                        with open(tgt_f_path, 'wb') as f:
                            pickle.dump(mask, f)  # H W


def complement_nusc_seg(part):
    # ----------------------------------------------------------
    # get sam id list to complement
    prompt = ['person', 'vehicle', 'road', 'traffic light']
    nusc_sensor='CAM_FRONT'
    nusc = NuScenes(version='v1.0-trainval', dataroot=os.path.join(dataset_root, 'nusc'), verbose=False)

    srcpath = os.path.join(dataset_root, 'nusc')
    instk_to_anntk_paths = [os.path.join(srcpath, 'extra/anns_train_ped_CAM_FRONT.pkl'),
                            os.path.join(srcpath, 'extra/anns_train_veh_CAM_FRONT.pkl'),
                            os.path.join(srcpath, 'extra/anns_val_ped_CAM_FRONT.pkl'),
                            os.path.join(srcpath, 'extra/anns_val_veh_CAM_FRONT.pkl'),
                            ]
    token_to_samid_path = os.path.join(srcpath, 'extra/token_id/trainval_token_to_sample_id.pkl')
    samid_to_token_path = os.path.join(srcpath, 'extra/token_id/trainval_sample_id_to_token.pkl')
    with open(token_to_samid_path, 'rb') as f:
        token_to_samid = pickle.load(f)
    with open(samid_to_token_path, 'rb') as f:
        samid_to_token = pickle.load(f)
    samids = []
    for path in instk_to_anntk_paths:
        with open(path, 'rb') as f:
            instk_to_anntks = pickle.load(f)
        for instk in instk_to_anntks:
            anntk_seqs = instk_to_anntks[instk]
            for seq in anntk_seqs:
                for anntk in seq:
                    ann = nusc.get('sample_annotation', anntk)
                    samid = token_to_samid[ann['sample_token']]
                    samids.append(samid)
    samidset1 = set(samids)

    samidset2 = set()
    segroot = '/home/y_feng/workspace6/datasets/nusc/extra/seg_sam/CAM_FRONT/person'
    for fnm in os.listdir(segroot):
        samid = fnm.replace('.pkl', '')
        samidset2.add(samid)
    
    samidlist = list(samidset1 - samidset2)
    print(samidlist)    
    n_parts = 4
    part_len = len(samidlist) // n_parts
    parts = [samidlist[i*part_len: (i+1)*part_len] for i in range(n_parts-1)]
    parts += [samidlist[(n_parts-1)*part_len:]]
    parts_len = 0
    for p in range(n_parts):
        parts_len += len(parts[p])
    assert len(samidlist) == parts_len

    cur_sam_id_list = parts[part]

    # ----------------------------------------------------------
    # original process
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = "./Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "./weights/groundingdino_swint_ogc.pth"

    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "./weights/sam_vit_h_4b8939.pth"

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(DEVICE)
    sam_predictor = SamPredictor(sam)


    # Predict classes and hyper-param for GroundingDINO
    CLASSES = [
               'person', 
               'vehicle', 
               'road', 
            #    'cross walks', 
               'traffic light',
            #    'sidewalks'
               ]
    
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.8

    # dataset setting
    DATASET_2_SRCPATH = {
        'TITAN': os.path.join(dataset_root, 'TITAN/honda_titan_dataset/dataset/images_anonymized'),
        'PIE': os.path.join(dataset_root, 'PIE_dataset/images'),
        'JAAD': os.path.join(dataset_root, 'JAAD/images'),
        'nuscenes': os.path.join(dataset_root, 'nusc'),
        'bdd100k': os.path.join(dataset_root, 'BDD100k/bdd100k/images/track')
    }
    DATASET_2_TGTPATH = {
        'TITAN': os.path.join(dataset_root, 'TITAN/TITAN_extra/seg_sam'),
        'PIE': os.path.join(dataset_root, 'PIE_dataset/seg_sam'),
        'JAAD': os.path.join(dataset_root, 'JAAD/seg_sam'),
        'nuscenes': os.path.join(dataset_root, 'nusc/extra/seg_sam/', nusc_sensor),
        'bdd100k': os.path.join(dataset_root, 'BDD100k/bdd100k/extra/seg_sam')
    }


    def seg_img(sam_predictor: SamPredictor, 
                image: np.ndarray, 
                ):
        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=BOX_THRESHOLD
        )

        # NMS post process
        # print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # print(f"After NMS: {len(detections.xyxy)} boxes")

        # Prompting SAM with detected boxes
        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, logits = sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )  # 每个cls预测多张mask，取score最高的那一张输出
                index = np.argmax(scores)
                result_masks.append(masks[index])
                # pdb.set_trace()
            # print(len(result_masks), xyxy.shape, image.shape)
            if len(result_masks) == 0:
                result_masks = np.zeros(shape=(len(prompt), image.shape[0], image.shape[1]))
            return np.array(result_masks)


        # convert detections to masks
        detec_masks = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )  # bool ndarray: n_detect,H,W
        
        # converge masks
        res = np.full((len(CLASSES),)+image.shape[:2], False)  # n_cls H W
        for i in range(len(detections.class_id)):
            try:
                cls = detections.class_id[i]
                res[cls] = res[cls]|detec_masks[i]
            except IndexError:
                pdb.set_trace()

        return res
    
    tgtpath = DATASET_2_TGTPATH['nuscenes']
    for samid in tqdm(cur_sam_id_list):
        samtk = samid_to_token[samid]
        sam = nusc.get('sample', samtk)
        sen_data = nusc.get('sample_data', sam['data'][nusc_sensor])
        img_path = os.path.join(srcpath, sen_data['filename'])
        src_img = cv2.imread(img_path)
        masks = seg_img(sam_predictor=sam_predictor,
                            image=src_img)  # n_cls H W
        for c in range(len(prompt)):
            cls = prompt[c].replace(' ', '_')
            mask = masks[c]
            makedir(os.path.join(tgtpath,
                                    cls))
            tgt_f_path = os.path.join(tgtpath, 
                                        cls, 
                                        samid+'.pkl')
            with open(tgt_f_path, 'wb') as f:
                pickle.dump(mask, f)

def get_ped_graph_seg(mode='ori_local',
                      target_size=(224, 224)):
    import argparse
    parser = argparse.ArgumentParser()
    # device
    parser.add_argument("--dataset_names", default='PIE_JAAD_TITAN_nuscenes_bdd100k', type=str)
    parser.add_argument("--oids_to_add_path", default=None, type=str)
    args = parser.parse_args()
    dataset_names = args.dataset_names
    oids_to_add = None
    if args.oids_to_add_path:
        with open(args.oids_to_add_path, 'rb') as f:
            oids_to_add = pickle.load(f)
    if 'PIE' in dataset_names:
        print('PIE')
        data_root = os.path.join(dataset_root, 'PIE_dataset')
        data_base = PIE(data_path=data_root)
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
        tgt_root = os.path.join(data_root, 'cropped_seg')
        src_root = os.path.join(data_root, 'seg_sam')
        tracks = data_base.generate_data_trajectory_sequence(image_set='all', 
                                                         **data_opts)  # all: 1842  train 882 
        num_tracks = len(tracks['image'])
        for i in tqdm(range(num_tracks)):
            cur_pid = tracks['ped_id'][i][0][0]  # [[id], [id], ...]
            setid, vidid, oid = cur_pid.split('_')  # str
            if oids_to_add is not None and oid not in oids_to_add:
                continue
            track_len = len(tracks['ped_id'][i])
            for j in range(track_len):
                img_path = tracks['image'][i][j]
                img_nm = img_path.split('/')[-1].replace('.png', '')
                l, t, r, b = tracks['bbox'][i][j]  # l t r b
                l, t, r, b = map(int, [l, t, r, b])
                if not os.path.exists(os.path.join(src_root, 'person', setid, vidid, img_nm+'.pkl')):
                    continue
                for cls in os.listdir(src_root):
                    src_path = os.path.join(src_root, 
                                            cls, 
                                            setid, 
                                            vidid, 
                                            img_nm+'.pkl')
                    with open(src_path, 'rb') as f:
                        segmap = pickle.load(f)  # H W int
                    segmap = np.stack([segmap]*3, axis=-1)*1  # H W 3 int
                    
                    cropped = crop_ctx(img=segmap, 
                                       bbox=[l, t, r, b],
                                       mode=mode,
                                       target_size=target_size,
                                       padding_value=0)
                    cropped = cropped[:, :, 0].astype(bool)  # H W bool
                    assert len(cropped.shape) == 2 and \
                        cropped.shape[0] == target_size[1] and\
                        cropped.shape[1] == target_size[0]
                    tgt_dir = os.path.join(tgt_root, 
                                           cls,
                                           mode, 
                                           str(target_size[0])+'w_by_'+str(target_size[1])+'h',
                                           setid,
                                           vidid,
                                           'ped',
                                           oid
                                           )
                    makedir(tgt_dir)
                    tgt_path = os.path.join(tgt_dir, img_nm+'.pkl')
                    with open(tgt_path, 'wb') as f:
                        pickle.dump(cropped, f)
    if 'JAAD' in dataset_names:
        print('JAAD')
        data_root = os.path.join(dataset_root, 'JAAD')
        data_base = JAAD(data_path=data_root)
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
        tgt_root = os.path.join(data_root, 'cropped_seg')
        src_root = os.path.join(data_root, 'seg_sam')
        tracks = data_base.generate_data_trajectory_sequence(image_set='all', 
                                                         **data_opts)  # all: 1842  train 882 
        num_tracks = len(tracks['image'])
        for i in tqdm(range(num_tracks)):
            cur_pid = tracks['ped_id'][i][0][0]  # [[id], [id], ...]
            _, vidid, oid = cur_pid.split('_')
            if oids_to_add is not None and oid not in oids_to_add:
                continue
            track_len = len(tracks['ped_id'][i])
            for j in range(track_len):
                img_path = tracks['image'][i][j]
                img_nm = img_path.split('/')[-1].replace('.png', '')
                if not os.path.exists(os.path.join(src_root, 'person', vidid, img_nm+'.pkl')):
                    continue
                for cls in os.listdir(src_root):
                    src_path = os.path.join(src_root, 
                                            cls, 
                                            vidid, 
                                            img_nm+'.pkl')
                    with open(src_path, 'rb') as f:
                        segmap = pickle.load(f)  # H W int
                    segmap = np.stack([segmap]*3, axis=-1)*1   # H W 3 int
                    l, t, r, b = tracks['bbox'][i][j]  # l t r b
                    l, t, r, b = map(int, [l, t, r, b])
                    cropped = crop_ctx(img=segmap, 
                                       bbox=[l, t, r, b],
                                       mode=mode,
                                       target_size=target_size,
                                       padding_value=0)
                    cropped = cropped[:, :, 0].astype(bool)  # H W bool
                    assert len(cropped.shape) == 2 and \
                        cropped.shape[0] == target_size[1] and\
                        cropped.shape[1] == target_size[0]
                    tgt_dir = os.path.join(tgt_root, 
                                           cls,
                                           mode, 
                                           str(target_size[0])+'w_by_'+str(target_size[1])+'h',
                                           vidid,
                                           'ped',
                                           oid
                                           )
                    makedir(tgt_dir)
                    tgt_path = os.path.join(tgt_dir, img_nm+'.pkl')
                    with open(tgt_path, 'wb') as f:
                        pickle.dump(cropped, f)
    if 'TITAN' in dataset_names:
        print('TITAN')
        titan = TITAN_dataset(sub_set='all')
        tracks = titan.p_tracks
        data_root = os.path.join(dataset_root, 'TITAN')
        src_root = os.path.join(data_root, 'TITAN_extra/seg_sam')
        tgt_root = os.path.join(data_root, 'TITAN_extra/cropped_seg')
        num_tracks = len(tracks['clip_id'])
        for i in tqdm(range(num_tracks)):
            vidid = int(tracks['clip_id'][i][0])
            oid = int(float(tracks['obj_id'][i][0]))
            if oids_to_add is not None and str(oid) not in oids_to_add:
                            continue
            for j in range(len(tracks['clip_id'][i])):  # time steps in each track
                img_nm = tracks['img_nm'][i][j].replace('.png', '')
                l, t, r, b = list(map(int, tracks['bbox'][i][j]))
                if not os.path.exists(os.path.join(src_root, 
                                                   'person', 
                                                   str(vidid), 
                                                   img_nm+'.pkl')):
                    continue
                for cls in os.listdir(src_root):
                    src_path = os.path.join(src_root, 
                                            cls, 
                                            str(vidid), 
                                            img_nm+'.pkl')
                    with open(src_path, 'rb') as f:
                        segmap = pickle.load(f)  # H W int
                    segmap = np.stack([segmap]*3, axis=-1)*1  # H W 3 int
                    cropped = crop_ctx(img=segmap, 
                                       bbox=[l, t, r, b],
                                       mode=mode,
                                       target_size=target_size,
                                       padding_value=0)
                    cropped = cropped[:, :, 0].astype(bool)  # H W bool
                    assert len(cropped.shape) == 2 and \
                        cropped.shape[0] == target_size[1] and\
                        cropped.shape[1] == target_size[0]
                    tgt_dir = os.path.join(tgt_root, 
                                           cls,
                                           mode, 
                                           str(target_size[0])+'w_by_'+str(target_size[1])+'h',
                                           str(vidid),
                                           'ped',
                                           str(oid)
                                           )
                    makedir(tgt_dir)
                    tgt_path = os.path.join(tgt_dir, img_nm+'.pkl')
                    with open(tgt_path, 'wb') as f:
                        pickle.dump(cropped, f)
    if 'nuscenes' in dataset_names:
        print('nuscenes')
        nusc = NuScenes(version='v1.0-trainval', 
                        dataroot=os.path.join(dataset_root, 'nusc'), 
                        verbose=True)
        with open(os.path.join(dataset_root, 
                               'nusc/extra/token_id/trainval_token_to_sample_id.pkl'),
                  'rb') as f:
            token_to_sample_id = pickle.load(f)
        with open(os.path.join(dataset_root, 
                               'nusc/extra/token_id/trainval_token_to_instance_id.pkl'),
                  'rb') as f:
            token_to_instance_id = pickle.load(f)
        with open(os.path.join(dataset_root, 
                               'nusc/extra/anns_train_ped_CAM_FRONT.pkl',
                               ),
                  'rb') as f:
            instk_to_anntks = pickle.load(f)
        with open(os.path.join(dataset_root, 
                               'nusc/extra/anns_val_ped_CAM_FRONT.pkl',
                               ),
                  'rb') as f:
            val_instk_to_anntks = pickle.load(f)
        instk_to_anntks.update(val_instk_to_anntks)
        data_root = os.path.join(dataset_root, 'nusc')
        src_root = os.path.join(data_root, 'extra/seg_sam/CAM_FRONT')
        tgt_root = os.path.join(data_root, 'extra/cropped_seg/CAM_FRONT')
        for instk in tqdm(instk_to_anntks):
            insid = token_to_instance_id[instk]
            anntk_seqs = instk_to_anntks[instk]
            seq_all = []
            for seq in anntk_seqs:
                seq_all += seq
            for anntk in seq_all:
                bbox = nusc_3dbbox_to_2dbbox(nusc, anntk)[0]  # ltrb
                l, t, r, b = bbox
                if r-l<2 or b-t<2:
                    continue
                ann = nusc.get('sample_annotation', anntk)
                samtk = nusc.get('sample', ann['sample_token'])['token']
                samid = token_to_sample_id[samtk]
                if not os.path.exists(os.path.join(src_root, 
                                                   'person', 
                                                   samid+'.pkl')):
                    continue
                for cls in os.listdir(src_root):
                    src_path = os.path.join(src_root, 
                                            cls,
                                            samid+'.pkl')
                    with open(src_path, 'rb') as f:
                        segmap = pickle.load(f)  # H W int
                    segmap = np.stack([segmap]*3, axis=-1)*1  # H W 3 int
                    cropped = crop_ctx(img=segmap, 
                                       bbox=[l, t, r, b],
                                       mode=mode,
                                       target_size=target_size,
                                       padding_value=0)
                    cropped = cropped[:, :, 0].astype(bool)  # H W bool
                    assert len(cropped.shape) == 2 and \
                        cropped.shape[0] == target_size[1] and\
                        cropped.shape[1] == target_size[0]
                    tgt_dir = os.path.join(tgt_root, 
                                           cls,
                                           mode, 
                                           str(target_size[0])+'w_by_'+str(target_size[1])+'h',
                                           'ped',
                                           insid,
                                           )
                    makedir(tgt_dir)
                    tgt_path = os.path.join(tgt_dir, samid+'.pkl')
                    with open(tgt_path, 'wb') as f:
                        pickle.dump(cropped, f)
    if 'bdd100k' in dataset_names:
        print('bdd100k')
        data_root = os.path.join(dataset_root, 'BDD100k/bdd100k')
        vid_nm2id_path = os.path.join(data_root, 'extra', 'vid_nm2id.pkl')
        with open(vid_nm2id_path, 'rb') as f:
            vid_nm2id = pickle.load(f)
        src_root = os.path.join(data_root, 'extra/seg_sam/')
        tgt_root = os.path.join(data_root, 'extra/cropped_seg')
        label_root = os.path.join(data_root, 'labels', 'box_track_20')
        sub_set='train_val'
        for _subset in reversed(sub_set.split('_')):
            print(f'Processing {_subset} set')
            label_dir = os.path.join(label_root, _subset)
            # traverse video
            for lfnm in tqdm(os.listdir(label_dir)):
                l_path = os.path.join(label_dir, lfnm)
                with open(l_path) as f:
                    vid_l = json.load(f)  # list
                vid_nm = lfnm.replace('.json', '')
                vid_id = vid_nm2id[vid_nm]
                # traverse label
                for img_l in vid_l:
                    img_nm = img_l['name']
                    img_id = img_nm.split('-')[-1].replace('.jpg', '')
                    img_id_int = int(img_id)
                    if not os.path.exists(os.path.join(src_root, 
                                                       'person',
                                                       str(vid_id),
                                                       str(img_id_int)+'.pkl')):
                        continue
                    for l in img_l['labels']:
                        cls = l['category']
                        oid = l['id']
                        oid = str(int(oid))
                        if oids_to_add is not None and oid not in oids_to_add:
                            continue
                        cls_k = 'ped' if cls in ('other person', 'pedestrian', 'rider') else 'veh'
                        if cls_k != 'ped':
                            continue
                        ltrb = [l['box2d']['x1'],
                                l['box2d']['y1'],
                                l['box2d']['x2'],
                                l['box2d']['y2']]
                        for cls in os.listdir(src_root):
                            src_path = os.path.join(src_root, 
                                                    cls,
                                                    str(vid_id),
                                                    str(img_id_int)+'.pkl')
                            with open(src_path, 'rb') as f:
                                segmap = pickle.load(f)  # H W int
                            segmap = np.stack([segmap]*3, axis=-1)*1  # H W 3 int
                            cropped = crop_ctx(img=segmap, 
                                                bbox=ltrb,
                                                mode=mode,
                                                target_size=target_size,
                                                padding_value=0)
                            cropped = cropped[:, :, 0].astype(bool)  # H W bool
                            assert len(cropped.shape) == 2 and \
                                cropped.shape[0] == target_size[1] and\
                                cropped.shape[1] == target_size[0]
                            tgt_dir = os.path.join(tgt_root, 
                                                    cls,
                                                    mode, 
                                                    str(target_size[0])+'w_by_'+str(target_size[1])+'h',
                                                    'ped',
                                                    str(int(oid)),
                                                    )
                            makedir(tgt_dir)
                            tgt_path = os.path.join(tgt_dir, str(img_id_int)+'.pkl')
                            with open(tgt_path, 'wb') as f:
                                pickle.dump(cropped, f)
                            
def complement_jaad_seg():
    prompt = ['person', 'vehicle', 'road', 'traffic light']
    vidid_list = ['67', '72', '84', '83', '62', '68', '79', '63', '73', '69', '78', '71', '80', '64', '75', '66', '70', '86', '76', '82', '81', '65', '77', '74', '85']
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default=0, type=int)
    args = parser.parse_args()
    part = args.part
    n_parts = 4
    part_len = len(vidid_list) // n_parts
    parts = [vidid_list[i*part_len: (i+1)*part_len] for i in range(n_parts-1)]
    parts += [vidid_list[(n_parts-1)*part_len:]]
    parts_len = 0
    for p in range(n_parts):
        parts_len += len(parts[p])
    assert len(vidid_list) == parts_len
    cur_part = parts[part]
    # ----------------------------------------------------------
    # original process
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = "./Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "./weights/groundingdino_swint_ogc.pth"

    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "./weights/sam_vit_h_4b8939.pth"

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(DEVICE)
    sam_predictor = SamPredictor(sam)


    # Predict classes and hyper-param for GroundingDINO
    CLASSES = [
               'person', 
               'vehicle', 
               'road', 
            #    'cross walks', 
               'traffic light',
            #    'sidewalks'
               ]
    
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.8

    # dataset setting
    DATASET_2_SRCPATH = {
        'TITAN': os.path.join(dataset_root, 'TITAN/honda_titan_dataset/dataset/images_anonymized'),
        'PIE': os.path.join(dataset_root, 'PIE_dataset/images'),
        'JAAD': os.path.join(dataset_root, 'JAAD/images'),
        'nuscenes': os.path.join(dataset_root, 'nusc'),
        'bdd100k': os.path.join(dataset_root, 'BDD100k/bdd100k/images/track')
    }
    DATASET_2_TGTPATH = {
        'TITAN': os.path.join(dataset_root, 'TITAN/TITAN_extra/seg_sam'),
        'PIE': os.path.join(dataset_root, 'PIE_dataset/seg_sam'),
        'JAAD': os.path.join(dataset_root, 'JAAD/seg_sam'),
        'nuscenes': os.path.join(dataset_root, 'nusc/extra/seg_sam/', 'CAM_FRONT'),
        'bdd100k': os.path.join(dataset_root, 'BDD100k/bdd100k/extra/seg_sam')
    }


    def seg_img(sam_predictor: SamPredictor, 
                image: np.ndarray, 
                ):
        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=BOX_THRESHOLD
        )

        # NMS post process
        # print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # print(f"After NMS: {len(detections.xyxy)} boxes")

        # Prompting SAM with detected boxes
        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, logits = sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )  # 每个cls预测多张mask，取score最高的那一张输出
                index = np.argmax(scores)
                result_masks.append(masks[index])
                # pdb.set_trace()
            # print(len(result_masks), xyxy.shape, image.shape)
            if len(result_masks) == 0:
                result_masks = np.zeros(shape=(len(prompt), image.shape[0], image.shape[1]))
            return np.array(result_masks)


        # convert detections to masks
        detec_masks = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )  # bool ndarray: n_detect,H,W
        
        # converge masks
        res = np.full((len(CLASSES),)+image.shape[:2], False)  # n_cls H W
        for i in range(len(detections.class_id)):
            try:
                cls = detections.class_id[i]
                res[cls] = res[cls]|detec_masks[i]
            except IndexError:
                pdb.set_trace()

        return res

    # traverse
    print('JAAD')
    srcpath = DATASET_2_SRCPATH['JAAD']
    tgtpath = DATASET_2_TGTPATH['JAAD']
    for v_id in tqdm(cur_part, desc='video loop'):
        v_path = os.path.join(srcpath, 
                              'video_'+ v_id.zfill(4))
        # v_id = clip.replace('video_', '')
        # v_id = str(int(v_id))
        tgt_v_dirs = []
        for cls in prompt:
            tgt_v_dir = os.path.join(tgtpath, cls.replace(' ', '_'), v_id)
            makedir(tgt_v_dir)
            tgt_v_dirs.append(tgt_v_dir)

        for img_nm in tqdm(os.listdir(v_path), desc='img loop'):
            src_img = cv2.imread(os.path.join(v_path, img_nm))
            img_id = img_nm.replace('.png', '')

            masks = seg_img(sam_predictor=sam_predictor,
                            image=src_img)  # 1 H W
            for c in range(len(prompt)):
                mask = masks[c]
                tgt_f_path = os.path.join(tgt_v_dirs[c],
                                            img_id+'.pkl')
                with open(tgt_f_path, 'wb') as f:
                    pickle.dump(mask, f)  # H W

def complement_ped_graph_seg_jaad(mode='ori_local',
                                target_size=(224, 224)):
    vidid_list = ['67', '72', '84', '83', '62', '68', '79', '63', '73', '69', '78', '71', '80', '64', '75', '66', '70', '86', '76', '82', '81', '65', '77', '74', '85']
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default=0, type=int)
    args = parser.parse_args()
    part = args.part
    n_parts = 4
    part_len = len(vidid_list) // n_parts
    parts = [vidid_list[i*part_len: (i+1)*part_len] for i in range(n_parts-1)]
    parts += [vidid_list[(n_parts-1)*part_len:]]
    parts_len = 0
    for p in range(n_parts):
        parts_len += len(parts[p])
    assert len(vidid_list) == parts_len
    cur_vidid_list = parts[part]
    print('JAAD')
    data_root = os.path.join(dataset_root, 'JAAD')
    data_base = JAAD(data_path=data_root)
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
    tgt_root = os.path.join(data_root, 'cropped_seg')
    src_root = os.path.join(data_root, 'seg_sam')
    tracks = data_base.generate_data_trajectory_sequence(image_set='all', 
                                                        **data_opts)  # all: 1842  train 882 
    num_tracks = len(tracks['image'])
    for i in tqdm(range(num_tracks)):
        cur_pid = tracks['ped_id'][i][0][0]  # [[id], [id], ...]
        _, vidid, oid = cur_pid.split('_')
        if not vidid in cur_vidid_list:
            continue
        track_len = len(tracks['ped_id'][i])
        for j in range(track_len):
            img_path = tracks['image'][i][j]
            img_nm = img_path.split('/')[-1].replace('.png', '')
            if not os.path.exists(os.path.join(src_root, 'person', vidid, img_nm+'.pkl')):
                continue
            for cls in os.listdir(src_root):
                src_path = os.path.join(src_root, 
                                        cls, 
                                        vidid, 
                                        img_nm+'.pkl')
                with open(src_path, 'rb') as f:
                    segmap = pickle.load(f)  # H W int
                segmap = np.stack([segmap]*3, axis=-1)*1   # H W 3 int
                l, t, r, b = tracks['bbox'][i][j]  # l t r b
                l, t, r, b = map(int, [l, t, r, b])
                cropped = crop_ctx(img=segmap, 
                                    bbox=[l, t, r, b],
                                    mode=mode,
                                    target_size=target_size,
                                    padding_value=0)
                cropped = cropped[:, :, 0].astype(bool)  # H W bool
                assert len(cropped.shape) == 2 and \
                    cropped.shape[0] == target_size[1] and\
                    cropped.shape[1] == target_size[0]
                tgt_dir = os.path.join(tgt_root, 
                                        cls,
                                        mode, 
                                        str(target_size[0])+'w_by_'+str(target_size[1])+'h',
                                        vidid,
                                        'ped',
                                        oid
                                        )
                makedir(tgt_dir)
                tgt_path = os.path.join(tgt_dir, img_nm+'.pkl')
                with open(tgt_path, 'wb') as f:
                    pickle.dump(cropped, f)
    


if __name__ == '__main__':
    # prompt = ('person', 'vehicle', 'road', 'traffic light')
    # segment_dataset(dataset_names='bdd100k',
    #                     prompt=prompt)

    # complement segmentation data for nusc
    get_ped_graph_seg()
    