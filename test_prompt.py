import sys
import os
os.environ['TORCHIO_BACKEND'] = 'soundfile'
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B')

# 使用你的录音测试
prompt_wav = './asset/prompt.wav'
prompt_text = '今天是星期一，要上班了，大家赶紧起床！'
tts_text = '你好，我是理枫'

print('Testing CosyVoice2-0.5B...')
for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=False, speed=1.0)):
    torchaudio.save('test_output_new.wav', j['tts_speech'], cosyvoice.sample_rate)
    duration = j['tts_speech'].shape[1] / cosyvoice.sample_rate
    print(f'Generated: {duration:.2f}s')
    print(f'Sample rate: {cosyvoice.sample_rate}')
    print(f'Shape: {j["tts_speech"].shape}')
    break
