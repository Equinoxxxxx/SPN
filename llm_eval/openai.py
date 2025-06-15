import openai
import anthropic
import cv2
import numpy as np
import os
import json
import argparse
import base64
import time
from tqdm import tqdm

from tools.utils import makedir
from tools.log import create_logger
from llm_eval.anthropic import anthropic_one_prompt
from llm_eval.prompts import beginning_prompt, example_prompt, abbr2modality
from llm_eval.keys import OPENAI, ANTHROPIC


def half_img_size(img_path: str):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    img = cv2.resize(img, (w//2, h//2))
    tmp_path = img_path.replace('.png', '_half.png')
    cv2.imwrite(tmp_path, img)
    return tmp_path


def load_base64_img(img_path: str):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def openai_gpt_one_prompt(prompt: str, 
                        img_path: str=None,
                        instructions: str=None,
                        model: str='gpt-4o',
                        temperature: float=0.0,
                        client: openai.OpenAI=None,
                        conversation_log: list=None):
    if client is None:
        client = openai.OpenAI(api_key=OPENAI)
    if conversation_log is None:
        conversation_log = []
    if img_path is not None:
        img_base64 = load_base64_img(img_path)
        conversation_log.append({"role": "user", 
                                 "content": [
                                            {"type": "input_text", 
                                             "text": prompt},
                                            {"type": "input_image", 
                                              "image_url": f"data:image/png;base64,{img_base64}",
                                              # 'detail': 'low'
                                              }
                                            ]})
    else:
        conversation_log.append({"role": "user", 
                                 "content": prompt})
    if 'o1' in model or 'o3' in model:
        response = client.responses.create(
          model=model,
          input=conversation_log,
          instructions=instructions,
        )
    else:
        response = client.responses.create(
            model=model,
            input=conversation_log,
            instructions=instructions,
            temperature=temperature,
        )
    conversation_log.append({"role": "assistant", 
                             "content": response.output_text})
    return response.output_text, conversation_log


def eval_explicitness(args):
    # init logger
    time_stamp = time.strftime("%d%b%Y-%Hh%Mm%Ss")
    log, logclose = create_logger(log_filename=os.path.join(args.explain_res_path, f'{args.model_name}_llm_eval_{time_stamp}.log'))
    # init client
    if 'gpt' in args.model_name or 'o1' in args.model_name or 'o3' in args.model_name:
        client = openai.OpenAI(api_key=OPENAI)
        promt_func = openai_gpt_one_prompt
    elif 'claude' in args.model_name:
        client = anthropic.Anthropic(api_key=ANTHROPIC)
        promt_func = anthropic_one_prompt
    else:
        raise ValueError(args.model_name)
    # init conversation log
    conversation_log = [
        {"role": "user", 
         "content": beginning_prompt},
    ]
    tmp_log = [
        {"role": "user", 
         "content": beginning_prompt},
    ]
    model_assess_txt = ''
    log(f'user: {beginning_prompt}')
    if args.use_example:
        conversation_log.append({"role": "user", 
                                 "content": example_prompt})
        tmp_log.append(
            {"role": "user", 
             "content": example_prompt}
        )
        log(f'user: {example_prompt}')
    # load samples
    sample_root = args.explain_res_path
    # proto loop
    for proto_id in tqdm(os.listdir(sample_root), desc='prototype id'):
        if not args.one_by_one:
            time.sleep(60)
        proto_path = os.path.join(sample_root, proto_id)
        if not os.path.isdir(proto_path):
            continue
        if not proto_id.isdigit():
            continue
        # sample loop
        for sample_id in tqdm(os.listdir(proto_path), desc='sample id'):
            sample_path = os.path.join(proto_path, sample_id)
            for m_abbr in os.listdir(sample_path):
                if m_abbr in ('ctx', 'sklt', 'traj', 'ego', 'social'):
                    modality_path = os.path.join(sample_path, m_abbr)
                    break
            modality = abbr2modality[m_abbr]
            if m_abbr == 'traj':
                img_path = os.path.join(modality_path, 'traj.png')
                img_path = half_img_size(img_path)
            elif m_abbr == 'social':
                img_path = os.path.join(modality_path, 'social.png')
                img_path = half_img_size(img_path)
            else:
                for f_nm in os.listdir(modality_path):
                    if f_nm.endswith('.png'):
                        img_path = os.path.join(modality_path, f_nm)
                        break
            proto_no = int(proto_id)+1
            sample_no = int(sample_id)+1
            prompt = f"This is sample {sample_no} of proto {proto_no}. It is in {modality} modality."
            try:
                output_txt, tmp_log = promt_func(prompt=prompt, 
                                                img_path=img_path, 
                                                model=args.model_name,
                                                temperature=args.temperature,
                                                client=client,
                                                conversation_log=tmp_log)
            except Exception as e:
                log(f'{e}')
                import pdb; pdb.set_trace()
                raise e
            
            log(f'{proto_no}-{sample_no}-{modality}\nuser: {prompt}\nmodel: {output_txt}')
            if args.one_by_one:
                conversation_log.append(tmp_log[-1])
        prompt = f'Now, all samples of prototype {proto_no} are shown. Please score how understandable the prototypes are from 0 to 10, and give a brief explanation for your score. Note that the number of modalities that represent in the samples does not affect the score.'
        output_txt, tmp_log = promt_func(prompt=prompt, 
                                            model=args.model_name,
                                            client=client,
                                            conversation_log=tmp_log)
        model_assess_txt += f'Prototype {proto_no}:\n{output_txt}\n'
        log(f'user: {prompt}\nmodel: {output_txt}')
        if args.one_by_one:
            conversation_log.append(tmp_log[-1])
            tmp_log = [
                {"role": "user", 
                 "content": beginning_prompt},
            ]
            if args.use_example:
                tmp_log.append(
                    {"role": "user", 
                    "content": example_prompt}
                )
    prompt = f'Now, all prototypes are shown. Please put all the scores in a list.'
    time.sleep(60)
    output_txt, conversation_log = promt_func(prompt=prompt, 
                                            model=args.model_name,
                                            client=client,
                                            conversation_log=conversation_log)
    log(f'user: {prompt}\nmodel: {output_txt}')
    prompt = f'The following is your assessment of all prototypes. Now compare the text explanations of these prototypes, ignore the scores you previously gave and give a new score for each prototype.'
    prompt += f'\n{model_assess_txt}'
    time.sleep(60)
    output_txt, conversation_log = promt_func(prompt=prompt, 
                                            model=args.model_name,
                                            client=client,
                                            conversation_log=conversation_log)
    log(f'user: {prompt}\nmodel: {output_txt}')
    logclose()
    return conversation_log


if __name__ == '__main__':
    from openai import OpenAI
    import os
    client = OpenAI(
        api_key = OPENAI
    )
    models = client.models.list()
    for model in models:
        print(model.id)