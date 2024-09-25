from ..datasets.PIE_JAAD import PIEDataset
from ..datasets.TITAN import TITAN_dataset
from ..datasets.nuscenes_dataset import NuscDataset
from ..datasets.bdd100k import BDD100kDataset
from ..datasets.identify_sample import get_ori_img_path
from ..visualize.visualize_bbox import draw_boxes_on_img
import os
import cv2
import pdb
from tqdm import tqdm
import numpy as np
import random


def get_bbox_from_tracks(tracks,
                          dataset_name='TITAN',
                          target_set_id=None,  # int
                          target_vid_id=None,  # int
                          target_oid=None,  # int
                          target_img_nm_int=None,
                          pie_v_track=False):
    if dataset_name == 'TITAN':
        n_track = len(tracks['clip_id'])
        for i in range(n_track):
            vid_id = tracks['clip_id'][i][0]  # str
            oid = tracks['obj_id'][i][0]  # str
            vid_id = int(vid_id)
            oid = int(oid)
            if oid == target_oid and vid_id == target_vid_id:
                l_track = len(tracks['img_nm_int'][i])
                for j in range(l_track):
                    img_nm_int = tracks['img_nm_int'][i][j]
                    if img_nm_int == target_img_nm_int:
                        return tracks['bbox'][i][j]
    elif dataset_name == 'PIE':
        # import pdb;pdb.set_trace()
        if pie_v_track:
            for sid in tracks:
                sid_int = int(sid)
                if target_set_id != sid_int:
                    continue
                for vid in tracks[sid]:
                    vid_int = int(vid)
                    if target_vid_id != vid_int:
                        continue
                    for oid in tracks[sid][vid]:
                        oid_int = int(oid)
                        if target_oid != oid_int:
                            continue
                        for i in range(len(tracks[sid][vid][oid]['img_nm'])):
                            for j in range(len(tracks[sid][vid][oid]['img_nm'][i])):
                                imgnm = tracks[sid][vid][oid]['img_nm'][i][j]
                                img_nm_int = int(imgnm.replace('.png', ''))
                                if target_img_nm_int == img_nm_int:
                                    return tracks[sid][vid][oid]['bbox'][i][j]
        else:
            n_track = len(tracks['image'])
            for i in range(n_track):
                set_id = int(tracks['ped_id'][i][0][0].split('_')[0])
                vid_id = int(tracks['ped_id'][i][0][0].split('_')[1])
                oid = int(tracks['ped_id'][i][0][0].split('_')[2])
                
                if oid == target_oid and set_id == target_set_id and vid_id == target_vid_id:
                    l_track  = len(tracks['image'][i])
                    for j in range(l_track):
                        img_nm = tracks['image'][i][j].split('/')[-1].split('.')[0]
                        img_nm_int = int(img_nm)
                        if img_nm_int == target_img_nm_int:
                            return tracks['bbox'][i][j]
    elif dataset_name == 'JAAD':
        n_track = len(tracks['image'])
        for i in range(n_track):
            set_id = int(tracks['ped_id'][i].split('_')[0][0])
            vid_id = int(tracks['ped_id'][i].split('_')[0][1])
            oid = int(tracks['ped_id'][i].split('_')[0][2])
            if oid == target_oid and vid_id == target_vid_id:
                l_track  = len(tracks['image'][i])
                for j in range(l_track):
                    img_nm = tracks['image'][i][j].split('/')[-1].split('.')[0]
                    img_nm_int = int(img_nm)
                    if img_nm_int == target_img_nm_int:
                        return tracks['bbox'][i][j]
    elif dataset_name == 'nuscenes':
        n_track = len(tracks['img_nm'])
        for i in range(n_track):
            oid = int(tracks['ins_id'][i][0])
            if oid == target_oid:
                l_track = len(tracks['sam_id'][i])
                for j in range(l_track):
                    try:
                        img_nm_int = int(tracks['sam_id'][i][j])
                    except:
                        import pdb;pdb.set_trace()
                        raise ValueError()
                    if img_nm_int == target_img_nm_int:
                        return tracks['bbox_2d'][i][j]
    elif dataset_name == 'bdd100k':
        n_track = len(tracks['img_id'])
        for i in range(n_track):
            vid_id = int(tracks['vid_id'][i][0])
            oid = int(tracks['obj_id'][i][0])
            if oid == target_oid:
                l_track = len(tracks['img_id'][i])
                for j in range(l_track):
                    img_nm_int = int(tracks['img_id_int'][i][j])
                    if img_nm_int == target_img_nm_int:
                        return tracks['bbox'][i][j]
    else:
        raise ValueError(f'dataset_name {dataset_name}')

def check_sample_bbox(obs_len = 4,
            pred_len = 4,
            overlap_ratio = 0.5,
            obs_fps = 2,
            ctx_format = 'ped_graph',
            small_set = 0,
            ):
    
    tte = [0, int((obs_len+pred_len+1)/obs_fps*30)] 

    titan = TITAN_dataset(sub_set='default_test',
                        obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap_ratio, 
                        obs_fps=obs_fps,
                        required_labels=[
                                            'atomic_actions', 
                                            'simple_context', 
                                            'complex_context', 
                                            'communicative', 
                                            'transporting',
                                            'age',
                                            ], 
                        multi_label_cross=0,  
                        use_cross=1,
                        use_atomic=1, 
                        use_complex=0, 
                        use_communicative=0, 
                        use_transporting=0, 
                        use_age=0,
                        tte=None,
                        modalities=['img','sklt','ctx','traj','ego', 'social'],
                        sklt_format='coord',
                        ctx_format=ctx_format,
                        augment_mode='none',
                        small_set=small_set,
                        max_n_neighbor=10,
                        )
    n_all_bbox = 0
    n_correct = 0
    n_wrong = 0
    for sample in tqdm(titan):
        vid_id = int(sample['vid_id_int'].item())
        oid = int(sample['ped_id_int'].item())
        for i in range(obs_len):
            bbox = sample['obs_bboxes'][i].detach().numpy().astype(int)  # 4
            img_nm_int = int(sample['img_nm_int'][i].item())
            bbox_from_track = get_bbox_from_tracks(tracks=titan.p_tracks,
                                                   dataset_name='TITAN',
                                                   target_vid_id=vid_id,
                                                   target_oid=oid,
                                                   target_img_nm_int=img_nm_int)
            try:
                bbox_from_track = np.array(bbox_from_track).astype(int)
            except:
                print(vid_id, oid, img_nm_int)
                import pdb; pdb.set_trace()
                raise ValueError()
            n_all_bbox += 1
            if sum(bbox-bbox_from_track) <= 4:
                n_correct += 1
            else:
                n_wrong += 1
                print(f'bbox different from track.\n bbox in sample{bbox} bbox in track{bbox_from_track}. oid: {oid} vid id: {vid_id} img nm {img_nm_int}')
    print(n_all_bbox, n_correct, n_wrong)

def check_neighbor_bbox(dataset_name='TITAN',
                        obs_len = 4,
                        pred_len = 4,
                        overlap_ratio = 0.5,
                        obs_fps = 2,
                        ctx_format = 'ped_graph',
                        small_set = 0,):
    tte = [0, int((obs_len+pred_len+1)/obs_fps*30)]
    if dataset_name == 'TITAN':
        dataset = TITAN_dataset(sub_set='default_test',
                            obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap_ratio, 
                            obs_fps=obs_fps,
                            required_labels=['atomic_actions', 
                                                'simple_context', 
                                                'complex_context', 
                                                'communicative', 
                                                'transporting',
                                                'age',
                                                ], 
                            multi_label_cross=0,  
                            use_cross=1,
                            use_atomic=1, 
                            use_complex=0, 
                            use_communicative=0, 
                            use_transporting=0, 
                            use_age=0,
                            tte=None,
                            modalities=['img','sklt','ctx','traj','ego', 'social'],
                            sklt_format='coord',
                            ctx_format=ctx_format,
                            augment_mode='none',
                            small_set=small_set,
                            max_n_neighbor=10,
                            )
    elif dataset_name == 'PIE':
        dataset = PIEDataset(dataset_name='PIE', seq_type='crossing',
                      obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap_ratio,
                      obs_fps=obs_fps,
                      do_balance=False, subset='train', bbox_size=(224, 224), 
                      img_norm_mode='torch', target_color_order='BGR',
                      resize_mode='even_padded', 
                      modalities=['img','sklt','ctx','traj','ego', 'social'],
                      sklt_format='coord',
                      ctx_format=ctx_format,
                      augment_mode='none',
                      tte=tte,
                      recog_act=0,
                      offset_traj=0,
                      speed_unit='m/s',
                      small_set=small_set,
                      )
    elif dataset_name == 'nuscenes':
        dataset = NuscDataset(sklt_format='coord',
                       modalities=['img','sklt','ctx','traj','ego', 'social'],
                       ctx_format=ctx_format,
                       small_set=small_set,
                       obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap_ratio,
                       obs_fps=obs_fps,
                       augment_mode='none',
                       min_h=2,
                       min_w=2,
                       min_vis_level=0
                       )
    elif dataset_name == 'bdd100k':
        dataset = BDD100kDataset(subsets='train',
                         sklt_format='coord',
                         modalities=['img','sklt','ctx','traj','ego', 'social'],
                         ctx_format=ctx_format,
                         small_set=small_set,
                         obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap_ratio,
                         obs_fps=obs_fps,
                         augment_mode='none',
                         min_h=2,
                         min_w=2,
                         )
    n_all_bbox = 0
    n_correct = 0
    n_wrong = 0
    for sample in tqdm(dataset):
        set_id = int(sample['set_id_int'].item())
        vid_id = int(sample['vid_id_int'].item())
        target_oid = int(sample['ped_id_int'].item())
        neighbor_relation = sample['obs_neighbor_relation'].detach().numpy().astype(int) # K T 5
        neighbor_cls = neighbor_relation[:,:,-1]
        neighbor_bbox = sample['obs_neighbor_bbox'].detach().numpy().astype(int) # K T 4
        neighbor_oid = sample['obs_neighbor_oid'].detach().numpy().astype(int) # K
        n_neighbor = len(neighbor_oid)
        for k in range(n_neighbor):
            oid = int(neighbor_oid[k])
            if oid == target_oid:
                import pdb;pdb.set_trace()
                raise ValueError()
            if oid < 0:
                continue
            cls = neighbor_relation[k, 0, -1]
            for i in range(obs_len):
                bbox = neighbor_bbox[k, i]  # 4
                if (bbox == 0).all():
                    continue
                img_nm_int = int(sample['img_nm_int'][i].item())
                if cls == 0:
                    bbox_from_track = get_bbox_from_tracks(tracks=dataset.p_tracks,
                                                        dataset_name=dataset_name,
                                                        target_set_id=set_id,
                                                        target_vid_id=vid_id,
                                                        target_oid=oid,
                                                        target_img_nm_int=img_nm_int)
                elif cls == 1:
                    bbox_from_track = get_bbox_from_tracks(tracks=dataset.v_tracks,
                                                        dataset_name=dataset_name,
                                                        target_set_id=set_id,
                                                        target_vid_id=vid_id,
                                                        target_oid=oid,
                                                        target_img_nm_int=img_nm_int,
                                                        pie_v_track=True)
                else:
                    continue
                if bbox_from_track is None:
                    n_all_bbox += 1
                    n_wrong += 1
                    print(f'bbox not found in tracks oid: {oid} vid id: {vid_id} img nm {img_nm_int} n all {n_all_bbox} n wrong {n_wrong}')
                    continue
                try:
                    bbox_from_track = np.array(bbox_from_track).astype(int)
                except:
                    oids = set()
                    n_track = len(dataset.p_tracks['image'])
                    for k in range(n_track):
                        if int(dataset.p_tracks['ped_id'][k][0][0].split('_')[0]) == set_id and int(dataset.p_tracks['ped_id'][k][0][0].split('_')[1]) == vid_id:
                            oids.add(int(dataset.p_tracks['ped_id'][k][0][0].split('_')[2]))
                    print(oids)
                    print(f'sid {set_id}, vid {vid_id}, oid {oid}, {img_nm_int}')
                    import pdb; pdb.set_trace()
                    raise ValueError()
                n_all_bbox += 1
                if sum(bbox-bbox_from_track) <= 4:
                    n_correct += 1
                else:
                    n_wrong += 1
                    print(f'bbox different from track.\n bbox in sample{bbox} bbox in track{bbox_from_track}. oid: {oid} vid id: {vid_id} img nm {img_nm_int}')
    print(n_all_bbox, n_correct, n_wrong)


def vis_neighbor_bbox(dataset_name='TITAN',
                      obs_len = 4,
                        pred_len = 4,
                        overlap_ratio = 0.5,
                        obs_fps = 2,
                        ctx_format = 'ped_graph',
                        small_set = 0,):
    
    
    tte = [0, int((obs_len+pred_len+1)/obs_fps*30)]
    if dataset_name == 'TITAN':
        dataset = TITAN_dataset(sub_set='default_test',
                            obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap_ratio, 
                            obs_fps=obs_fps,
                            required_labels=['atomic_actions', 
                                                'simple_context', 
                                                'complex_context', 
                                                'communicative', 
                                                'transporting',
                                                'age',
                                                ], 
                            multi_label_cross=0,  
                            use_cross=1,
                            use_atomic=1, 
                            use_complex=0, 
                            use_communicative=0, 
                            use_transporting=0, 
                            use_age=0,
                            tte=None,
                            modalities=['img','sklt','ctx','traj','ego', 'social'],
                            sklt_format='coord',
                            ctx_format=ctx_format,
                            augment_mode='none',
                            small_set=small_set,
                            max_n_neighbor=10,
                            )
    if dataset_name == 'PIE':
        dataset = PIEDataset(dataset_name='PIE', seq_type='crossing',
                      obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap_ratio,
                      obs_fps=obs_fps,
                      do_balance=False, subset='train', bbox_size=(224, 224), 
                      img_norm_mode='torch', target_color_order='BGR',
                      resize_mode='even_padded', 
                      modalities=['img','sklt','ctx','traj','ego', 'social'],
                      sklt_format='coord',
                      ctx_format=ctx_format,
                      augment_mode='none',
                      tte=None,
                      recog_act=0,
                      offset_traj=0,
                      speed_unit='m/s',
                      small_set=small_set,
                      )
    if dataset_name == 'nuscenes':
        dataset = NuscDataset(sklt_format='coord',
                       modalities=['img','sklt','ctx','traj','ego', 'social'],
                       ctx_format=ctx_format,
                       small_set=small_set,
                       obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap_ratio,
                       obs_fps=obs_fps,
                       augment_mode='none',
                       min_h=2,
                       min_w=2,
                       min_vis_level=0
                       )
    if dataset_name == 'bdd100k':
        dataset = BDD100kDataset(subsets='train',
                         sklt_format='coord',
                         modalities=['img','sklt','ctx','traj','ego', 'social'],
                         ctx_format=ctx_format,
                         small_set=small_set,
                         obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap_ratio,
                         obs_fps=obs_fps,
                         augment_mode='none',
                         min_h=2,
                         min_w=2,
                         )
    t = random.randint(0, obs_len-1)
    idx = random.randint(0, len(dataset)-1)
    sample = dataset[idx]
    set_id = int(sample['set_id_int'].item())
    vid_id = int(sample['vid_id_int'].item())
    img_nm = sample['img_nm_int'][t].item()
    print('img nms', sample['img_nm_int'])
    img_path = get_ori_img_path(dataset,
                            set_id=set_id,
                            vid_id=vid_id,
                            img_nm=img_nm)
    img = cv2.imread(img_path)
    neighbor_bbox = sample['obs_neighbor_bbox'].detach().numpy()
    neighbor_cls = sample['obs_neighbor_relation'][:,:,-1].detach().numpy()  # K T
    neighbor_oid = sample['obs_neighbor_oid'].detach().numpy()  # K
    target_bbox = sample['obs_bboxes_unnormed'].detach().numpy()
    oid = sample['ped_id_int'].detach().numpy()
    new_img = draw_boxes_on_img(img,
                                neighbor_bbox[:, t],
                                interval=1,
                                ids=neighbor_oid)
    new_img = draw_boxes_on_img(new_img,
                                target_bbox,
                                color='r',
                                interval=1)
    print(target_bbox.shape, target_bbox)
    print(neighbor_bbox[:, t].shape, neighbor_bbox[:, t])
    print('neighbor cls', neighbor_cls[:, t].shape, neighbor_cls[:, t])
    print(f'neighbor oid {neighbor_oid}')

    import pdb;pdb.set_trace()
    cv2.imwrite(f'tools/test/neighbor_{dataset_name}_idx_{idx}_t_{t}.png', new_img)
    print(f'sample idx {idx} t {t}')

if __name__ == '__main__':
    # check_neighbor_bbox(dataset_name='nuscenes')
    vis_neighbor_bbox(dataset_name='bdd100k')