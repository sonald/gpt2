from lightning_whisper_mlx import LightningWhisperMLX
import sys


whisper = LightningWhisperMLX(model="large-v3", batch_size=1, quant=None)

if len(sys.argv) > 1:
    audio = sys.argv[1]
else:
    raise "need filename"
text = whisper.transcribe(audio_path=audio)['text']

print(text)# Use a pipeline as a high-level helper