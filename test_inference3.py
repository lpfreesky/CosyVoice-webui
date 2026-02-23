import sys
import os
os.environ['TORCHIO_BACKEND'] = 'soundfile'
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

# 使用绝对路径
model_dir = r'C:\Users\ZhuanZ1\.openclaw\workspace\CosyVoice\pretrained_models\CosyVoice2-0.5B'

print('Loading CosyVoice2-0.5B model from:', model_dir)
cosyvoice = AutoModel(model_dir=model_dir)
print('Model loaded!')
print('Sample rate:', cosyvoice.sample_rate)

# 简单的 zero-shot 测试
print('Testing inference...')
prompt_wav = './asset/zero_shot_prompt.wav'
for i, j in enumerate(cosyvoice.inference_zero_shot('你好，我是理枫的AI助手。', '希望你以后能够做的比我还好呦。', prompt_wav)):
    torchaudio.save('test_output.wav', j['tts_speech'], cosyvoice.sample_rate)
    print('Saved test_output.wav')
    break
print('Done!')
