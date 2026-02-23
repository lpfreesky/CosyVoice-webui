import sys
import os
os.environ['TORCHIO_BACKEND'] = 'soundfile'
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

model_dir = r'C:\Users\ZhuanZ1\.openclaw\workspace\CosyVoice\pretrained_models\CosyVoice2-0.5B'
cosyvoice = AutoModel(model_dir=model_dir)

# 测试文本
tts_text = '我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。'
prompt_text = '希望你以后能够做的比我还好呦。'
prompt_wav = './asset/prompt.wav'

print('Testing with CosyVoice2-0.5B...')
print(f'tts_text: {tts_text}')
print(f'prompt_text: {prompt_text}')

for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=False, speed=1.0)):
    torchaudio.save('test_length.wav', j['tts_speech'], cosyvoice.sample_rate)
    duration = j['tts_speech'].shape[1] / cosyvoice.sample_rate
    print(f'Generated audio duration: {duration:.2f} seconds')
    break
