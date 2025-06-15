import anthropic
import cv2
import numpy as np
import os
import json
import argparse
import base64
from tqdm import tqdm

from tools.utils import makedir
from tools.log import create_logger
from llm_eval.prompts import beginning_prompt, example_prompt, abbr2modality
from llm_eval.keys import ANTHROPIC


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
    
def anthropic_one_prompt(prompt: str, 
                            img_path: str=None,
                            instructions: str=None,
                            model: str='claude-3-7-sonnet-latest',
                            temperature: float=0.0,
                            client: anthropic.Anthropic=None,
                            conversation_log: list=None):
    if client is None:
        client = anthropic.Anthropic(api_key=ANTHROPIC)
    if conversation_log is None:
        conversation_log = []
    if img_path is not None:
        img_base64 = load_base64_img(img_path)
        conversation_log.append({"role": "user", 
                                 "content": [
                                            {"type": "text", 
                                             "text": prompt},
                                            {"type": "image",
                                             "source": {
                                                        "type": "base64",
                                                        "media_type": "image/png",
                                                        "data": img_base64,
                                                        },
                                            }
                                            ]})
    else:
        conversation_log.append({"role": "user", 
                                 "content": prompt})
    if model == 'claude':
        model = 'claude-3-7-sonnet-latest'
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=conversation_log,
        temperature=temperature,
    )
    conversation_log.append({"role": "assistant", 
                             "content": response.content[0].text})
    return response.content[0].text, conversation_log



if __name__ == '__main__':
    from llm_eval.prompts import example_path
    client = anthropic.Anthropic(api_key=ANTHROPIC)
    print(client.models.list(limit=20))
    # image1_data = load_base64_img(example_path)
    # image1_media_type = "image/png"
    # message = client.messages.create(
    #     model="claude-3-7-sonnet-20250219",
    #     max_tokens=1024,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "image",
    #                     "source": {
    #                         "type": "base64",
    #                         "media_type": "image/png",
    #                         "data": image1_data,
    #                     },
    #                 },
    #                 {
    #                     "type": "text",
    #                     "text": "Describe this image."
    #                 }
    #             ],
    #         }
    #     ],
    # )
    # print(message.content[0].text)
    # print(type(message.content[0].text))