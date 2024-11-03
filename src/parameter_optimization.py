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

# more accurate with spectrogram
y_normalized, rms, rms_dB_mean, g = ia.normalize_audio(y)

#could use gain=g to scale the dynamic range
dyn_rms = ia.compute_dynamic_rms(rms, gain=g)

input.mean_rms = np.rint(rms_dB_mean)
input.dynamic_range = dyn_rms

print(f"RMS: {input.mean_rms} and dynamic range: {input.dynamic_range}\
 rounded to nearest integer and gain coeff {g}")

"""
def on_model(model: Model) -> None:
    print(f"{model}")

ctl = Control()

ctl.load("parameter_guessing.lp")

ctl.gound()

result = ctl.solve(on_model=on_model)

print(result)
"""