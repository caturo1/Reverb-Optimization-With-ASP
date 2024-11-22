import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import numpy.typing as npt

RATE = 44_100

def normalize_signal(sig: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Normalization based on the root mean square energy."""

    rms = np.sqrt(np.mean(sig**2))
    return sig / rms if rms > 0 else sig


def load_audio(file: str) -> Tuple[np.ndarray[np.float32],int]:
    """Convert Samples to dBFS."""

    y, sr = librosa.load(path=file, sr=RATE, mono=False)
    return to_stereo(y), sr


def to_stereo(
        y: Optional[np.ndarray]
        ) -> Tuple[np.ndarray[np.float32],np.ndarray[np.float32]]:
    """Create stereo signal by duplicating the mono signal if necessary"""

    mono = y.ndim == 1 or y.shape[0] == 1
    if not mono:
        return y
    return np.vstack((y,y))


def compute_STFT(
        y: Optional[np.ndarray], 
        sr: float
        ) -> Tuple[np.ndarray[np.float32],np.ndarray[np.float32]]:
    """
    Computation of the STFT for both stereo channels

    To compute the 
    - Magnitude spectrum we use np.abs(S)
    - Phase spectrum we use np.angle(S)
    n_fft is adapted to a sample rate 22_050 but we use a sample rate
    of 44_100 and thus double the window size for proper frequency resolution
    """

    S_left = librosa.stft(y[0], n_fft=2048*2, hop_length=512)
    S_right = librosa.stft(y[1], n_fft=2048*2, hop_length=512)
    return S_left, S_right

def asp_scaling(arr):
    scaler = MinMaxScaler(feature_range=(0,100))
    return scaler.fit_transform(arr)

def rms_features(
        y: Optional[np.ndarray]
        ) -> Tuple[np.ndarray[np.float32], int, int, int]:
    """
    Root mean square value calculation of stereo channels.
    
    A central features that provides a measure for the energy contained
    in the signal. The result is scaled on [0,100] so that ASP can 
    handle the values and we can transform the results into
    corresponding reverb parameters.
    """

    rms = librosa.feature.rms(y=librosa.to_mono(y))[0]
    rms_left = librosa.feature.rms(y=y[0])[0]
    rms_right = librosa.feature.rms(y=y[1])[0]
    scaled_rms = asp_scaling(rms.reshape(-1, 1))
    scaled_left = asp_scaling(rms_left.reshape(-1, 1))
    scaled_right = asp_scaling(rms_right.reshape(-1, 1))
    rms_mean = np.rint(np.mean(scaled_rms))
    rms_left_mean = np.rint(np.mean(scaled_left))
    rms_right_mean = np.rint(np.mean(scaled_right))
    rms_channel_balance = np.rint(np.abs(rms_left_mean - rms_right_mean))
    return scaled_rms, rms_mean, rms_channel_balance


def compute_dynamic_rms(scaled_rms: np.ndarray[np.float32]) -> int:
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


def mean_spectral_centroid(
        y: Optional[np.ndarray], 
        sr: float
        ) -> Tuple[float,float]:
    """
    Center of gravity of spectral energy

    This gives us an idea about the brightness of the input
    and ultimetely hints for the room size and damping.
    """
    
    spec_cen_left = librosa.feature.spectral_centroid(y=y[0]+1e-10, sr=sr, n_fft=2048*2)[0]
    spec_cen_right = librosa.feature.spectral_centroid(y=y[1]+1e-10, sr=sr, n_fft=2048*2)[0]
    mean_spectral_centroid = np.rint(np.mean([spec_cen_left, spec_cen_right]))
    return mean_spectral_centroid, spec_cen_left, spec_cen_right

def mean_spectral_flatness(
        y: Optional[np.ndarray]
        ) -> Tuple[float,float]:
    """
    Noisiness vs Tonalness
    
    This gives us hints about dry/wet settings and damping
    """
    flatness_left = librosa.feature.spectral_flatness(y=y[0])
    flatness_right = librosa.feature.spectral_flatness(y=y[1])
    mean_flatness = 0.5 * (np.mean(flatness_left) + np.mean(flatness_right))
    return np.rint(mean_flatness)

def spectral_spread(
        S: Optional[np.ndarray],
        sr: float,
        centroid_left: np.ndarray,
        centroid_right: np.ndarray
        ) -> int:
    """
    Instantaneous bandwidth

    Hints at timbre by describing how stationary the sound is.
    Used for getting a better spectral resolution of the input.
    """

    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048*2)
    S_left_mag = np.abs(S[0])
    S_right_mag = np.abs(S[1])

    S_left_norm = S_left_mag / (np.sum(S_left_mag, axis=0, keepdims=True) + 1e-10)
    S_right_norm = S_right_mag / (np.sum(S_right_mag, axis=0, keepdims=True) + 1e-10)

    spread_left = np.sqrt(np.sum(((freqs.reshape(-1,1) - centroid_left)**2) * S_left_norm, axis=0))
    spread_right = np.sqrt(np.sum(((freqs.reshape(-1,1) - centroid_right)**2) * S_right_norm, axis=0))
    
    return int(np.mean([np.mean(spread_left), np.mean(spread_right)]))

def compute_spectral_rolloff(
        y: Optional[np.ndarray],
        sr: float
        ) -> Tuple[np.ndarray[np.float32],np.ndarray[np.float32]]:
    """ Indication about audio bandwith in low/high freqs

    Might be redundant concidering we use spectral centroid
    and spectral flatness.    
    """
    rolloff_left = librosa.feature.spectral_rolloff(y=y[0], sr=sr)[0]
    rolloff_right = librosa.feature.spectral_rolloff(y=y[1], sr=sr)[0]
    return rolloff_left, rolloff_right


def mid_side(y: Optional[np.ndarray]) ->Tuple[np.ndarray[np.float32],np.ndarray[np.float32]]:
    """Mid/side analysis for stereo spread parameter."""
    sum = (y[0] + y[1]) / 2
    diff = (y[0] - y[1]) / 2
    scaled_sum = asp_scaling(sum.reshape(-1,1))
    scaled_diff = asp_scaling(diff.reshape(-1,1))
    mid = np.rint(np.mean(scaled_sum))
    side = np.rint(np.mean(scaled_diff))
    return mid, side


# TODO transient detection (not sure yet, onset detection/COG/librosa peak_pick/flux)
# Transients may lead to changes of sound quality when manipulating the signal.

# zero crossing as indication for noisiness in time domain 
# redundant if spectral flatness is used and less informative since high tonal components receive high values as well)
def noisyness(y):
    return np.mean(librosa.feature.zero_crossing_rate(y))