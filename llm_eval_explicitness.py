import openai
import cv2
import numpy as np
import os
import json
import argparse

from tools.utils import makedir
from llm_eval.openai import openai_gpt_one_prompt, eval_explicitness
from llm_eval.keys import OPENAI
from llm_eval.prompts import beginning_prompt, example_prompt, proto_example_paths


def llm_eval():
    parser = argparse.ArgumentParser(description='main args')
    parser.add_argument("--model_name", default='gpt-4o', type=str)
    parser.add_argument("--use_example", default=0, type=int)
    parser.add_argument("--one_by_one", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--explain_res_path", 
                        default='/home/y_feng/workspace6/work/PedSpace/exp_dir/pedspace_sklt_ctx_traj_ego_social/exp717/explain/stage1_e20/relative_var', 
                        type=str)
    parser.add_argument("--calc_correlation", default=0, type=int)
    args = parser.parse_args()
    # path
    root = args.explain_res_path
    log_dir = os.path.join(root, 'llm_eval'+args.model_name)
    makedir(log_dir)
    conversation_log = eval_explicitness(args)
    # save conversation log
    with open(os.path.join(log_dir, 'conversation_log.json'), 'w') as f:
        json.dump(conversation_log, f)

if __name__ == '__main__':
    llm_eval()