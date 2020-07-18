import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
from scipy.io.wavfile import write

# Setup hparams

hparams = create_hparams()
hparams.sampling_rate = 22050

# Load model from checkpoint
checkpoint_path = "./models/tacotron2_statedict.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

# Load WaveGlow for mel2audio synthesis and denoiser

waveglow_path = './models/waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

# Prepare input text

text = "Waveglow is really awesome!"
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()

# Decode text input
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
audio_path = "./audio/audio.wav"
write(audio_path, hparams.sampling_rate, audio.cpu().numpy())


