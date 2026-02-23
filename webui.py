# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['3s极速复刻']
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def generate_audio(tts_text, prompt_text, prompt_wav_upload, seed, speed):
    prompt_wav = prompt_wav_upload if prompt_wav_upload is not None else None
    
    if prompt_wav is None:
        gr.Warning('请上传prompt音频文件！')
        yield (cosyvoice.sample_rate, default_data)
        return
        
    if prompt_text == '':
        gr.Warning('请输入prompt文本！')
        yield (cosyvoice.sample_rate, default_data)
        return
    
    logging.info('get zero_shot inference request')
    set_all_random_seed(seed)
    all_speech = []
    for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=False, speed=speed, text_frontend=False):
        all_speech.append(i['tts_speech'].numpy().flatten())
    
    if all_speech:
        yield (cosyvoice.sample_rate, np.concatenate(all_speech))
    else:
        yield (cosyvoice.sample_rate, default_data)


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### CosyVoice2-0.5B 语音克隆\n使用说明：\n1. 上传你的参考音频（录音）\n2. 在prompt文本框输入音频中相同的内容\n3. 输入要合成的文本\n4. 点击生成音频")
        
        with gr.Row():
            with gr.Column():
                tts_text = gr.Textbox(label="要合成的文本", lines=2, placeholder="输入你想要AI说的话...")
            with gr.Column():
                prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label="参考音频(你的录音)",)
        
        with gr.Row():
            prompt_text = gr.Textbox(label="参考音频的文本内容", lines=1, placeholder="必须与录音内容完全一致！")
        
        with gr.Row():
            seed = gr.Number(value=0, label="随机种子(0=随机)")
            speed = gr.Number(value=1.0, label="语速(0.5-2.0)", minimum=0.5, maximum=2.0, step=0.1)
            generate_button = gr.Button("生成音频", variant="primary")

        audio_output = gr.Audio(label="合成结果")

        generate_button.click(generate_audio,
                              inputs=[tts_text, prompt_text, prompt_wav_upload, seed, speed],
                              outputs=[audio_output])
    demo.queue(max_size=2, default_concurrency_limit=1)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B')
    args = parser.parse_args()
    
    logging.info('Loading CosyVoice2-0.5B model...')
    cosyvoice = AutoModel(model_dir=args.model_dir)
    logging.info('Model loaded!')
    
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    
    main()
