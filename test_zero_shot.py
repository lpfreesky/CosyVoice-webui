import sys
import os
os.environ['TORCHIO_BACKEND'] = 'soundfile'
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

# 使用绝对路径
model_dir = r'C:\Users\ZhuanZ1\.openclaw\workspace\CosyVoice\pretrained_models\CosyVoice2-0.5B'
# 使用转换后的 wav 文件
prompt_wav = r'C:\Users\ZhuanZ1\Documents\录音\prompt.wav'

print('Loading model...')
cosyvoice = AutoModel(model_dir=model_dir)
print('Model loaded!')
print('Sample rate:', cosyvoice.sample_rate)

# 检查音频文件
print('Checking prompt audio...')
try:
    audio_info = torchaudio.info(prompt_wav)
    print('Audio sample rate:', audio_info.sample_rate)
    print('Audio channels:', audio_info.num_channels)
    print('Audio frames:', audio_info.num_frames)
except Exception as e:
    print('Error reading audio:', e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3s极速复刻测试
print('\nTesting 3s zero-shot inference...')
prompt_text = "你好，我是理枫的AI助手。"
tts_text = "希望你以后能够做得更好。"

try:
    for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav)):
        torchaudio.save('test_output.wav', j['tts_speech'], cosyvoice.sample_rate)
        print('Saved test_output.wav')
        break
    print('Done!')
except Exception as e:
    print('Error during inference:')
    import traceback
    traceback.print_exc()
