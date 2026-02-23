import sys
import os
os.environ['TORCHIO_BACKEND'] = 'soundfile'
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

model_dir = r'C:\Users\ZhuanZ1\.openclaw\workspace\CosyVoice\pretrained_models\CosyVoice2-0.5B'
cosyvoice = AutoModel(model_dir=model_dir)

# 测试文本 - 短句
tts_text = '你好世界'
prompt_text = '希望你以后能够做的比我还好呦。'
prompt_wav = './asset/prompt.wav'

print('Test 1: Short text')
for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=False, speed=1.0)):
    duration = j['tts_speech'].shape[1] / cosyvoice.sample_rate
    print(f'Short text duration: {duration:.2f} seconds')
    break

# 测试原始长文本
tts_text2 = '我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。'

print('Test 2: Long text')
for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text2, prompt_text, prompt_wav, stream=False, speed=1.0)):
    duration = j['tts_speech'].shape[1] / cosyvoice.sample_rate
    print(f'Long text duration: {duration:.2f} seconds')
    break
