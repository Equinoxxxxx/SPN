from ..datasets.PIE_JAAD import PIEDataset
from ..datasets.TITAN import TITAN_dataset
from ..datasets.nuscenes_dataset import NuscDataset
from ..datasets.bdd100k import BDD100kDataset
from ..datasets.identify_sample import get_img_path
from ..visualize import draw_boxes_on_img
import os
import cv2


obs_len = 4
pred_len = 4
overlap_ratio = 0.5
obs_fps = 2
ctx_format = 'ped_graph'
small_set = 0
tte = [0, int((obs_len+pred_len+1)/obs_fps*30)] 

titan = TITAN_dataset(sub_set='default_test', norm_traj=1,
                      obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap_ratio, 
                      obs_fps=obs_fps,
                      required_labels=[
                                        'atomic_actions', 
                                        'simple_context', 
                                        'complex_context', 
                                        'communicative', 
                                        'transporting',
                                        'age'
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
                      augment_mode='random_crop_hflip',
                      small_set=small_set,
                      )

pie = PIEDataset(dataset_name='PIE', seq_type='crossing',
                  obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap_ratio,
                  obs_fps=obs_fps,
                  do_balance=False, subset='train', bbox_size=(224, 224), 
                  img_norm_mode='torch', color_order='BGR',
                  resize_mode='even_padded', 
                  modalities=['img','sklt','ctx','traj','ego', 'social'],
                  sklt_format='coord',
                  ctx_format=ctx_format,
                  augment_mode='random_crop_hflip',
                  tte=tte,
                  recog_act=0,
                  normalize_pos=0,
                  speed_unit='m/s',
                  small_set=small_set,
                  )

nusc = NuscDataset(sklt_format='coord',
                   modalities=['img','sklt','ctx','traj','ego', 'social'],
                   ctx_format=ctx_format,
                   small_set=small_set,
                   obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap_ratio,
                   obs_fps=obs_fps,)
bdd = BDD100kDataset(subsets='train',
                     sklt_format='coord',
                     modalities=['img','sklt','ctx','traj','ego', 'social'],
                     ctx_format=ctx_format,
                     small_set=small_set,
                     obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap_ratio,
                     obs_fps=obs_fps,)

titan_img_root = '/home/y_feng/workspace6/datasets/TITAN/honda_titan_dataset/dataset/images_anonymized'
img_root = titan_img_root
t = -1
sample = titan[t]
cid = str(sample['vid_id_int'].item())
img_nm = sample['img_nm_int'][-1].item()
img_path = get_img_path(titan,
                        vid_id=cid,
                        img_nm=img_nm)
img = cv2.imread(img_path)
neighbor_bbox = sample['obs_neighbor_bbox'].detach().numpy()
new_img = draw_boxes_on_img(img,
                            neighbor_bbox[:, t])
cv2.imwrite(new_img, 'test/neighbor.png')