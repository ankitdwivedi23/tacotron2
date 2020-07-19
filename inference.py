import matplotlib
import matplotlib.pylab as plt
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

def plot_data(data, filepath, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')
    fig.savefig(filepath)

# Setup hparams

print("Setting up hyperparams...")

hparams = create_hparams()
hparams.sampling_rate = 22050

print("Done.")

# Load model from checkpoint
checkpoint_path = "./models/tacotron2_statedict.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()
#tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
#tacotron2 = tacotron2.to('cuda')
#tacotron2.eval()


# Load WaveGlow for mel2audio synthesis and denoiser

waveglow_path = './models/waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']

'''
waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
waveglow.eval()
'''

waveglow.cuda().eval().half()   
#for k in waveglow.convinv:
#    k.float()
#denoiser = Denoiser(waveglow)


# Prepare input text
text = "Waveglow is really awesome!"
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()

# Decode text input and plot results
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
plot_data((mel_outputs.float().data.cpu().numpy()[0],
           mel_outputs_postnet.float().data.cpu().numpy()[0],
           alignments.float().data.cpu().numpy()[0].T), './melplots/plot.png')


with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

audio_path = "./audio/audio_custom_custom_halfprecision.wav"

write(audio_path, hparams.sampling_rate, audio[0].cpu().numpy())

'''
text = "Waveglow is really awesome!"
# preprocessing
sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)

# run the models
with torch.no_grad():
    _, mel, _, _ = tacotron2.infer(sequence)
    audio = waveglow.infer(mel)

audio_numpy = audio[0].data.cpu().numpy()
rate = 22050

audio_path = "./audio/audio.wav"
write(audio_path, rate, audio_numpy)
'''
