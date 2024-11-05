import os
from clingo.control import Control
from clingo import Model
import numpy as np
import feature_extraction.audio_features as audio_features
import input_analysis as ia

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the audio file
filename = os.path.join(script_dir, "../data/schreihals.wav")

input = audio_features.AudioFeatures()

y, sr = ia.load_audio(filename)
S = ia.compute_STFT(y=y, sr=sr)
# more accurate with spectrogram
rms, rms_mean = ia.rms_features(y)
dyn_rms = ia.compute_dynamic_rms(rms)
mean_spectral_centroid = ia.mean_spectral_centroid(y, sr)

input.mean_rms = rms_mean
input.dynamic_range = dyn_rms
# Density of the input audio based on inverse relationship of dr and rms
input.density = (100 - dyn_rms) * rms_mean
# The higher the spectral centroid, the higher the daming should be


print(f"Scaled RMS: {input.mean_rms} amd Scaled Dynamic Range: {input.dynamic_range}\n")

"""
def on_model(model: Model) -> None:
    print(f"{model}")

ctl = Control()

ctl.load("parameter_guessing.lp")

ctl.gound()

result = ctl.solve(on_model=on_model)

print(result)
"""