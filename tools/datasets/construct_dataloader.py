import torch
from tools.datasets.TITAN import TITAN_dataset
from tools.datasets.PIE_JAAD import PIEDataset
from tools.datasets.nuscenes_dataset import NuscDataset
from tools.datasets.bdd100k import BDD100kDataset

def construct_data_loader(args, log=print):
    datasets = [
        {
        'train':{k:None for k in args.dataset_names[0]},
        'val':{k:None for k in args.test_dataset_names[0]},
        'test':{k:None for k in args.test_dataset_names[0]},
        },
        {
        'train':{k:None for k in args.dataset_names[1]},
        'val':{k:None for k in args.test_dataset_names[1]},
        'test':{k:None for k in args.test_dataset_names[1]},
        },
    ]
    for stage in range(len(datasets)):
        for subset in datasets[stage]:
            _subset = subset
            _overlap = args.overlap1 if stage == 0 else args.overlap2
            _small_set = args.test_small_set
            if subset == 'pre':
                _subset = 'train'
                _small_set = args.small_set
            elif subset == 'train':
                _small_set = args.small_set
            for name in datasets[stage][subset]:
                if name == 'TITAN':
                    cur_set = TITAN_dataset(sub_set='default_'+_subset, 
                                            offset_traj=False,
                                            img_norm_mode=args.img_norm_mode, 
                                            target_color_order=args.model_color_order,
                                            obs_len=args.obs_len, 
                                            pred_len=args.pred_len, 
                                            overlap_ratio=_overlap, 
                                            obs_fps=args.obs_fps,
                                            recog_act=False,
                                            multi_label_cross=False, 
                                            act_sets=args.act_sets,
                                            loss_weight='sklearn',
                                            small_set=_small_set,
                                            resize_mode=args.resize_mode, 
                                            modalities=args.modalities,
                                            img_format=args.img_format,
                                            sklt_format=args.sklt_format,
                                            ctx_format=args.ctx_format,
                                            traj_format=args.traj_format,
                                            ego_format=args.ego_format,
                                            social_format=args.social_format,
                                            augment_mode=args.augment_mode,
                                            max_n_neighbor=args.max_n_neighbor,
                                            )
                if name in ('PIE', 'JAAD'):
                    cur_set = PIEDataset(dataset_name=name, 
                                         seq_type='crossing',
                                        subset=_subset,
                                        obs_len=args.obs_len, 
                                        pred_len=args.pred_len, 
                                        overlap_ratio=_overlap, 
                                        obs_fps=args.obs_fps,
                                        do_balance=False, 
                                        bbox_size=(224, 224), 
                                        img_norm_mode=args.img_norm_mode, 
                                        target_color_order=args.model_color_order,
                                        resize_mode=args.resize_mode,
                                        modalities=args.modalities,
                                        img_format=args.img_format,
                                        sklt_format=args.sklt_format,
                                        ctx_format=args.ctx_format,
                                        traj_format=args.traj_format,
                                        ego_format=args.ego_format,
                                        social_format=args.social_format,
                                        small_set=_small_set,
                                        tte=args.tte,
                                        recog_act=False,
                                        offset_traj=False,
                                        augment_mode=args.augment_mode,
                                            max_n_neighbor=args.max_n_neighbor,)
                    if subset in ('test', 'val'):
                        cur_set.tte = args.test_tte
                if name == 'nuscenes':
                    cur_set = NuscDataset(subset=_subset,
                                        obs_len=args.obs_len, 
                                        pred_len=args.pred_len, 
                                        overlap_ratio=_overlap, 
                                        obs_fps=args.obs_fps,
                                        small_set=_small_set,
                                        augment_mode=args.augment_mode,
                                        resize_mode=args.resize_mode,
                                        target_color_order=args.model_color_order, 
                                        img_norm_mode=args.img_norm_mode,
                                        modalities=args.modalities,
                                        img_format=args.img_format,
                                        sklt_format=args.sklt_format,
                                        ctx_format=args.ctx_format,
                                        traj_format=args.traj_format,
                                        ego_format=args.ego_format,
                                        social_format=args.social_format,
                                            max_n_neighbor=args.max_n_neighbor,
                                        )
                if name == 'bdd100k':
                    cur_set = BDD100kDataset(subsets=_subset,
                                            obs_len=args.obs_len, 
                                            pred_len=args.pred_len, 
                                            overlap_ratio=_overlap, 
                                            obs_fps=args.obs_fps,
                                            target_color_order=args.model_color_order, 
                                            img_norm_mode=args.img_norm_mode,
                                            small_set=_small_set,
                                            resize_mode=args.resize_mode,
                                            modalities=args.modalities,
                                            img_format=args.img_format,
                                            sklt_format=args.sklt_format,
                                            ctx_format=args.ctx_format,
                                            traj_format=args.traj_format,
                                            ego_format=args.ego_format,
                                            social_format=args.social_format,
                                            augment_mode=args.augment_mode,
                                            max_n_neighbor=args.max_n_neighbor,
                                            )
                datasets[stage][subset][name] = cur_set    
    for stage in range(len(datasets)):
        for _sub in datasets[stage]:
            for nm in datasets[stage][_sub]:
                if datasets[stage][_sub][nm] is not None:
                    log(f'{_sub} {nm} {len(datasets[stage][_sub][nm])}')
    
    concat_train_sets = []
    val_sets = []
    test_sets = []
    train_loaders = []
    val_loaders = []
    test_loaders = []
    for stage in range(len(datasets)):
        concat_train_sets.append(
            torch.utils.data.ConcatDataset(
                [datasets[stage]['train'][k] for k in datasets[stage]['train']]
            )
        )
        val_sets.append([datasets[stage]['val'][k] for k in datasets[stage]['val']])
        test_sets.append([datasets[stage]['test'][k] for k in datasets[stage]['test']])
        _batch_size = args.batch_size[stage]
        train_loaders.append(
            torch.utils.data.DataLoader(concat_train_sets[stage],
                                        batch_size=_batch_size, 
                                        shuffle=args.shuffle,
                                        num_workers=args.dataloader_workers,
                                        pin_memory=True,
                                        drop_last=True)
        )
        val_loaders.append(
            [torch.utils.data.DataLoader(val_sets[stage][i], 
                                        batch_size=_batch_size, 
                                        shuffle=args.shuffle,
                                        num_workers=args.dataloader_workers,
                                        pin_memory=True,
                                        drop_last=True
                                        ) for i in range(len(val_sets[stage]))]
        )
        test_loaders.append(
            [torch.utils.data.DataLoader(test_sets[stage][i], 
                                        batch_size=_batch_size, 
                                        shuffle=args.shuffle,
                                        num_workers=args.dataloader_workers,
                                        pin_memory=True,
                                        drop_last=True
                                        ) for i in range(len(test_sets[stage]))]
        )
    return train_loaders, val_loaders, test_loaders