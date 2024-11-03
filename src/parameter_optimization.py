import os
from clingo.control import Control
from clingo import Model
import numpy as np
import audio_features
import input_analysis as ia

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the audio file
filename = os.path.join(script_dir, "../data/schreihals.wav")

input = audio_features.AudioFeatures()

y, sr = ia.load_audio(filename)
# only calculate if I need it later, but for now it's fine (room for optimization)
S = ia.compute_STFT(y, sr)

# more accurate with spectrogram
rms = ia.compute_rms(S)
rms_dB = 20 * np.log10(rms)

dyn_rms = ia.compute_dynamic_rms(rms)

input.mean_rms = np.mean(rms)
input.dynamic_range = dyn_rms

print(input.mean_rms, input.dynamic_range)

def on_model(model: Model) -> None:
    """    
        Print the current model to the console
    """ 
    print(f"{model}")

ctl = Control()

ctl.load("parameter_guessing.lp")

ctl.gound()

result = ctl.solve(on_model=on_model)

print(result)