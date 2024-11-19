import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import numpy.typing as npt

CHUNK = 2048
CHANNELS = 2
RATE = 44_100
BIT_DEPTH = 16

# normalization based on the root mean square energy
def normalize_signal(sig):
    rms = np.sqrt(np.mean(sig**2))
    return sig / rms if rms > 0 else sig

# seperate function to extract the sampling rate alone
def check_sr(name) -> int:
    return librosa.get_samplerate(name)

# conversion of rms to dB
def rms_to_db(rms) -> float:
    return np.abs(30 + 20 * np.log10(rms))

def load_audio(file) -> Tuple[np.NDArray[np.float32],int]:
    # Convert Samples to dBFS
    y, sr = librosa.load(path=file, sr=RATE, mono=False)
    return y, sr

# create stereo signal by duplicating the mono signal
def to_stereo(y) -> Tuple[np.NDArray[np.float32],np.NDArray[np.float32]]:
    mono = y.ndim == 1 or y.shape[0] == 1
    if not mono: 
        return 
    return np.vstack((y,y))

# To compute the 
# - Magnitude spectrum we use np.abs(S)
# - Phase spectrum we use np.angle(S)
def compute_STFT(y, sr) -> Tuple[np.NDArray[np.float32],np.NDArray[np.float32]]:
    S_left = librosa.stft(y[0])
    S_right = librosa.stft(y[1])
    return S_left, S_right

# center of gravity of spectral energy
def mean_spectral_centroid(y, sr) -> Tuple[float,float]:
    # at silent parts, high frequency components might dominate,
    # therefore add constant term
    spec_cen_left = librosa.feature.spectral_centroid(y=y[0]+1e-10, sr=sr)[0]
    spec_cen_right = librosa.feature.spectral_centroid(y=y[1]+1e-10, sr=sr)[0]
    mean_spectral_centroid = np.mean(np.mean(spec_cen_left) + np.mean(spec_cen_right))
    return mean_spectral_centroid

# measures noisiness vs tonalness
def mean_spectral_flatness(y) -> Tuple[float,float]:
    flatness_left = librosa.feature.spectral_flatness(y=y[0])
    flatness_right = librosa.feature.spectral_flatness(y=y[1])
    mean_flatness_left = np.mean(flatness_left)
    mean_flatness_right = np.mean(flatness_right)
    return mean_flatness_left, mean_flatness_right

# not in use yet
def compute_spectral_rolloff(y, sr) -> Tuple[np.NDArray[np.float32],np.NDArray[np.float32]]:
    rolloff_left = librosa.feature.spectral_rolloff(y=y[0], sr=sr)[0]
    rolloff_right = librosa.feature.spectral_rolloff(y=y[1], sr=sr)[0]
    return rolloff_left, rolloff_right

# TODO transient detection (not sure yet, onset detection/COG/librosa peak_pick/flux)
# Transients may lead to changes of sound quality when manipulating the signal.

# zero crossing as indication for noisiness in time domain 
# redundant if spectral flatness is used and less informative since high tonal components receive high values as well)
def noisyness(y):
    return np.mean(librosa.feature.zero_crossing_rate(y))

# mid/side analysis for stereo spread parameter 
def mid_side(y, is_mono=False) ->Tuple[np.NDArray[np.float32],np.NDArray[np.float32]]:
    mid = 0.5 * (y[0] + y[1])
    side = 0.5 * (y[0] - y[1])
    
    return mid, side

def rms_to_dB(rms, eps=1e-20):
    return librosa.power_to_db(rms**2)

# window_size = n_fft size (default 2048)
# not sure, if the scaling is done properly or of it distorts the feature perception
# scaling, such that ASP can deal with the values
def rms_features(y) -> Tuple[npt.NDArray[np.float32], int]:
    scaler = MinMaxScaler(feature_range=(0,100))
    rms = librosa.feature.rms(y=y[0])
    scaled_rms = scaler.fit_transform(rms[0].reshape(-1, 1))
    rms_mean = np.rint(np.mean(scaled_rms))
    return scaled_rms, rms_mean

# percentile calculation to get an estimate in order to avoid clips and nosie distorting the dynamic range
def compute_dynamic_rms(scaled_rms) -> int:
    rms = np.ravel(scaled_rms)
    if len(rms) == 0:
        return 0
    p97 = np.percentile(rms, 97)  
    p3 = np.percentile(rms, 3)    
    
    return np.rint(p97 - p3)
