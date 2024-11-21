import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import numpy.typing as npt

RATE = 44_100

def normalize_signal(sig):
    """Normalization based on the root mean square energy."""

    rms = np.sqrt(np.mean(sig**2))
    return sig / rms if rms > 0 else sig


def load_audio(file) -> Tuple[np.ndarray[np.float32],int]:
    """Convert Samples to dBFS."""

    y, sr = librosa.load(path=file, sr=RATE, mono=False)
    return to_stereo(y), sr


def to_stereo(y) -> Tuple[np.ndarray[np.float32],np.ndarray[np.float32]]:
    """Create stereo signal by duplicating the mono signal if necessary"""

    mono = y.ndim == 1 or y.shape[0] == 1
    if not mono: 
        return y
    return np.vstack((y,y))


def compute_STFT(y, sr) -> Tuple[np.ndarray[np.float32],np.ndarray[np.float32]]:
    """
    Computation of the STFT for both stereo channels

    To compute the 
    - Magnitude spectrum we use np.abs(S)
    - Phase spectrum we use np.angle(S)
    """

    S_left = librosa.stft(y[0])
    S_right = librosa.stft(y[1])
    return S_left, S_right


def rms_features(y) -> Tuple[np.ndarray[np.float32], int]:
    """
    Root mean square value calculation of stereo channels.
    
    A central features that provides a measure for the energy contained
    in the signal. The result is scaled on [0,100] so that ASP can 
    handle the values and we can transform the results into
    corresponding reverb parameters.
    """

    scaler = MinMaxScaler(feature_range=(0,100))
    rms = librosa.feature.rms(y=librosa.to_mono(y))
    rms_left = librosa.feature.rms(y=y[0])
    rms_right = librosa.feature.rms(y=y[1])
    scaled_rms = scaler.fit_transform(rms[0].reshape(-1, 1))
    rms_mean = np.rint(np.mean(scaled_rms))
    return scaled_rms, rms_mean, rms_left, rms_right


def compute_dynamic_rms(scaled_rms) -> int:
    """
    Dynamic range calculation based on rms features

    Calculation is done with percentiles to avoid amplitude outliers (ie. clips)
    to distort the dynamic range result.
    """

    rms = np.ravel(scaled_rms)
    if len(rms) == 0:
        return 0
    p97 = np.percentile(rms, 97)  
    p3 = np.percentile(rms, 3)    
    
    return np.rint(p97 - p3)


def mean_spectral_centroid(y, sr) -> Tuple[float,float]:
    """
    Center of gravity of spectral energy

    This gives us an idea about the brightness of the input
    and ultimetely hints for the room size and damping.
    """
    
    spec_cen_left = librosa.feature.spectral_centroid(y=y[0]+1e-10, sr=sr)[0]
    spec_cen_right = librosa.feature.spectral_centroid(y=y[1]+1e-10, sr=sr)[0]
    mean_spectral_centroid = 0.5 * (np.mean(spec_cen_left) + np.mean(spec_cen_right))
    return np.rint(mean_spectral_centroid)

def mean_spectral_flatness(y) -> Tuple[float,float]:
    """
    Noisiness vs Tonalness
    
    This gives us hints about dry/wet settings and damping
    """
    flatness_left = librosa.feature.spectral_flatness(y=y[0])
    flatness_right = librosa.feature.spectral_flatness(y=y[1])
    mean_flatness = 0.5 * (np.mean(flatness_left) + np.mean(flatness_right))
    return np.rint(mean_flatness)

def compute_spectral_rolloff(y, sr) -> Tuple[np.ndarray[np.float32],np.ndarray[np.float32]]:
    """ Indication about audio bandwith in low/high freqs

    Might be redundant concidering we use spectral centroid
    and spectral flatness.    
    """
    rolloff_left = librosa.feature.spectral_rolloff(y=y[0], sr=sr)[0]
    rolloff_right = librosa.feature.spectral_rolloff(y=y[1], sr=sr)[0]
    return rolloff_left, rolloff_right


def mid_side(y) ->Tuple[np.ndarray[np.float32],np.ndarray[np.float32]]:
    """Mid/side analysis for stereo spread parameter."""
    mid = 0.5 * (y[0] + y[1])
    side = 0.5 * (y[0] - y[1])
    return mid, side


# TODO transient detection (not sure yet, onset detection/COG/librosa peak_pick/flux)
# Transients may lead to changes of sound quality when manipulating the signal.

# zero crossing as indication for noisiness in time domain 
# redundant if spectral flatness is used and less informative since high tonal components receive high values as well)
def noisyness(y):
    return np.mean(librosa.feature.zero_crossing_rate(y))