beginning_prompt = 'You are an annotation expert. You are now given a task to evaluate the explainability of a deep learning model. The model is trained to predict the future actions of pedestrians based on a set of prototypes. Each prototype is trained to fire on a certain pattern of features such as the presence of the crosswalk, the layout of the trajectories, etc. Each prototype is represented by 6 representative samples. These samples are in the following 5 modalities: 1) context, which is the image of the scene that surrounds the pedestrian; 2) skeleton, which is the body key points of the pedestrian, overlayed by the heatmap that highlights the selient joints; 3) trajectory, which is the bounding box trajectory of the pedestrian during the past 2 seconds, with dark color for the distant past and light color for the recent past; 4) ego motion, which is the acceleration of the ego vehicle, plotted in red; 5) social interaction, which decribes the spatial relations between the pedestrian and the other agents in the scene, with the target pedestrian marked by a red hollow bounding box and the neighbor agents marked by transparent colored bounding boxes (warm colors for the salient neighbors). After all samples of a prototype are shown, please score how understandable the prototypes are from 0 to 10, and give a brief explanation for your score. Note that the samples that represent a prototype might not include all the modalities.'
abbr2modality = {
        'ctx': 'context',
        'sklt': 'skeleton',
        'traj': 'trajectory',
        'ego': 'ego motion',
        'social': 'social interaction',
    }


proto_example_paths = [
        # only providing scores for 10 and 1
        # 378 6 83.23 crosswalk
        '/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp378/explain_epoch_24/6',
        # 726 46 47.126 small region on pedestrian leg
        '/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp726/explain_stage_1_epoch_22/activation/46',
        # 708 2 101.071 small region on pedestrian lowerbody
        '/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp708/explain_stage_1_epoch_40/activation/2',
        # 726 0 45.935 small strip region on pedestrian body
        '/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp726/explain_stage_1_epoch_22/activation/0',
        # 717 29 63.817 trajectory moving from left edge to right
        '/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp717/explain/stage1_e20/relative_var/29',
        # 717 23 99.326 target ped left bottom corner
        '/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp717/explain/stage1_e20/relative_var/23',
        # 717 47 117.81 target ped on top of the truck
        '/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp717/explain/stage1_e20/relative_var/47',
        # 726 33 4.645 smoothly decreasing positive acceleration
        '/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp726/explain_stage_1_epoch_22/activation/33',
        # 711 26 4.371 trajectory moving from center to right
        '/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp711/explain_stage_1_epoch_20/activation/26',
        # 711 28 2.215 social salient big van
        '/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp711/explain_stage_1_epoch_20/activation/28',
        # 711 16 2.206 target ped relatively right, no salient neighbor
        '/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp711/explain_stage_1_epoch_20/activation/16',
        # 726 35 4.687  meaningless scatter in context
        '/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp726/explain_stage_1_epoch_22/activation/35',
        # 517 13 15.414 strip heatmap
        '/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp517/explain/stage1_e20/13',
    ]

example_prompt = [
    
]

example_path = '/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp726/explain_stage_1_epoch_22/activation/0/1/ctx/mean_max_t.png'

epochs = {
    '726': 22,
    '708': 2,
    '711': 20,
    '717': 20,
}

proto_rank_topkms = {
    '726': [40,46,21,20,0,4,30,19,44,24,34,7,13,45,31,37,36,9,5,47,16,38,12,35,33,8,25,15,28,26,
            41,29,49,18,6,23,32,48,1,2,3,10,11,14,17,22,27,39,42,43],
    '708': [29,31,39,25,24,40,48,18,30,9,36,1,38,42,47,3,26,0,7,19,45,14,6,16,27,37,23,32,13,33,
            44,4,2,5,46,8,10,11,12,15,17,20,21,22,28,34,35,41,43,49],
    '711': [1,39,42,2,17,40,13,24,35,37,11,33,8,4,19,23,10,31,20,7,44,34,15,27,49,9,14,47,48,38,
            0,30,12,41,3,21,36,26,46,32,43,18,29,5,28,16,25,45,6,22],
    '717': [22,38,7,1,47,5,23,25,15,40,11,3,2,28,6,29,39,13,21,35,24,46,26,32,18,43,4,10,17,44,
            42,49,0,30,45,8,20,48,14,27,12,37,31,41,16,36,33,9,19,34]
}



llm_eval_log = {
    '726': {
        'claude': {
            'claude_llm_eval.log':[
                8,8,8,8,6,8,9,8,9,7,
                8,8,8,8,8,8,7,8,7,7,
                8,7,8,8,7,8,8,8,9,6,
                7,8,8,10,8,8,8,6,8,8,
                8,8,8,8,7,7,8,8,9,8,
                ],
            'claude_llm_eval_08Apr2025-16h48m52s.log':[
                7,9,9,9,6,7,10,6,10,8,
                9,9,8,8,9,8,8,9,8,7,
                8,7,8,8,7,7,9,8,9,5,
                8,7,8,10,7,8,7,7,8,8,
                6,8,9,8,7,7,8,7,10,8,
                ],
            },
            'gpt-4o': {
                'gpt-4o_llm_eval_07Apr2025-23h52m46s.log': [
                    5,4,3,4,3,5,2,4,2,4,
                    4,4,7,3,5,6,5,3,7,3,
                    3,3,3,7,3,5,4,4,9,6,
                    4,6,7,1,3,5,5,3,4,4,
                    3,4,4,3,3,3,4,6,2,4,
                    ]
                },
            'gpt-4.5-preview': {
                'gpt-4.5-preview_llm_eval_08Apr2025-00h21m17s.log': [
                    3,3,4,3,3,5,1,3,2,4,
                    3,4,4,4,4,8,3,3,3,3,
                    4,3,4,7,3,2,4,4,2,5,
                    3,3.5,4,2,3,4.5,4,3.5,3,3,
                    3,3,3,3,3,3,4,3,2,3,
                    ]
                }
            
        },
    '717': {
        'claude': {
                'claude_llm_eval_10Apr2025-18h44m34s.log': [
                    6,10,9,8,7,8,8,8,5,7,
                    7,8,7,9,7,9,6,7,7,7,
                    8,8,9,8,8,8,8,7,9,8,
                    7,7,8,7,8,7,6,7,6,9,
                    9,7,7,7,7,7,9,7,7,8,
                ],
                'claude-3-5-sonnet-latest_llm_eval_10Apr2025-20h12m44s.log': [
                    9,10,9,9,9,9,9,9,8,7,
                    8,9,9,9,8,9,9,8,9,8,
                    9,8,9,8,9,9,9,7,9,9,
                    8,8,9,8,8,8,8,7,7,9,
                    9,7,9,9,8,7,9,9,8,9,
                ],
            },
        'gpt-4o': {
                'gpt-4o_llm_eval_10Apr2025-19h55m40s.log': [
                    5,3,7,6,5,6,6,7,5,4,
                    5,6,5,7,5,6,5,5,5,4,
                    5,6,4,5,6,6,5,4,5,6,
                    6,6,3,5,6,7,5,5,5,5,
                    6,5,5,4,6,6,6,5,5,5,
                ]
            },
        }
    
        }

llm_eval_res = {
    '726': {
        'claude': {
            'claude_llm_eval.log':[
                8,8,8,8,6,8,9,8,9,7,
                8,8,8,8,8,8,7,8,7,7,
                10,10,8,8,7,8,8,8,9,6,
                7,8,8,10,8,8,8,6,8,8,
                10,8,8,8,7,7,10,8,10,8,
                ],
            'claude_llm_eval_08Apr2025-16h48m52s.log':[
                7,9,9,9,6,7,10,6,10,8,
                9,9,8,8,9,8,8,9,8,7,
                8,7,8,8,7,7,9,8,9,5,
                8,7,8,10,7,8,7,7,8,8,
                6,8,9,8,7,7,8,7,10,8,
                ],
            'claude_llm_eval_08Apr2025-16h48m52s_.log':[
                9,7,7,7,7,7,6,6,8,8,
                9,9,8,7,8,8,6,7,8,7,
                8,9,8,8,7,7,7,8,8,5,
                8,7,8,9,7,6,7,7,8,8,
                9,8,7,8,7,7,9,7,7,8,
                ],
            },
        'gpt-4o': {
                'gpt-4o_llm_eval_07Apr2025-23h52m46s.log': [
                    5,4,3,4,3,5,2,4,2,4,
                    4,4,7,3,5,6,5,3,7,3,
                    3,7,3,7,3,5,4,4,9,6,
                    4,6,7,1,3,5,5,3,4,4,
                    7,4,4,3,3,3,7,6,2,4,
                    ]
                },
        'gpt-4.5-preview': {
                'gpt-4.5-preview_llm_eval_08Apr2025-00h21m17s.log': [
                    6,3,4,3,5.5,5,1,4,2,4,
                    3,4,4,4,4,7,3,3,3,3,
                    6,6,4,5,3,2,4,4,2,5,
                    3,3.5,4,2,3,4.5,4,3.5,3,3,
                    8.5,3,3,3,3,3,7,3,2,3,
                    ]
                },
        },
    '717': {
        'claude': {
                'claude_llm_eval_10Apr2025-18h44m34s.log': [
                    6,10,9,8,7,8,8,8,5,7,
                    7,8,7,9,7,9,6,7,7,7,
                    8,8,9,8,8,8,8,7,9,8,
                    7,7,8,7,8,7,6,7,6,9,
                    9,7,7,7,7,7,9,7,7,8,
                ],
                'claude-3-5-sonnet-latest_llm_eval_10Apr2025-20h12m44s.log': [
                    9,10,9,9,9,9,9,9,8,7,
                    8,9,9,9,8,9,9,8,9,8,
                    9,8,9,8,9,9,9,7,9,9,
                    8,8,9,8,8,8,8,7,7,9,
                    9,7,9,9,8,7,9,9,8,9,
                ],
            },
        'gpt-4o': {
                'gpt-4o_llm_eval_10Apr2025-19h55m40s.log': [
                    5,3,7,6,5,6,6,7,5,4,
                    5,6,5,7,5,6,5,5,5,4,
                    5,6,4,5,6,6,5,4,5,6,
                    6,6,3,5,6,7,5,5,5,5,
                    6,5,5,4,6,6,6,5,5,5,
                ]
            },
        },
}
