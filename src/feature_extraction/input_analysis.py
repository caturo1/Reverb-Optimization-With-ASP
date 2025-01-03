import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional

RATE = 44_100
NFFT = 2048*2
HOPS = 512*2
SQRT_2 = np.sqrt(2)

def normalize_signal(sig: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Normalization based on the root mean square energy."""

    rms = np.sqrt(np.mean(sig**2))
    return sig / rms if rms > 0 else sig

def rms_to_db(rms) -> float:
    """Just to compare with iZotope results"""

    return 20 * np.log10(rms + 1e-10)

def load_audio(file: str) -> Tuple[np.ndarray[np.float32],int]:
    """Load audio input as a stereo file"""
    y, sr = librosa.load(path=file, sr=RATE, mono=False)
    return to_stereo(y), sr
    
def to_stereo(
        y: Optional[np.ndarray]
        ) -> Tuple[np.ndarray[np.float32],np.ndarray[np.float32]]:
    """Create stereo signal by duplicating the mono signal if necessary"""
    if y is None:
        return ValueError("Input signal cannot be None")

    mono = y.ndim == 1 or y.shape[0] == 1
    if not mono:
        return y
    return np.vstack((y,y))


def compute_STFT(
        y: np.ndarray, 
        mode: str,
        sr: float = RATE
        ) -> Tuple[np.ndarray[np.float32],np.ndarray[np.float32]]:
    """
    Computation of the STFT for both stereo channels

    To compute the 
    - Magnitude spectrum we use np.abs(S)
    - Phase spectrum we use np.angle(S)
    n_fft is adapted to a sample rate 22_050 but we use a sample rate
    of 44_100 and thus double the window size for proper frequency resolution
    """
    if (mode == "regular"):
        S_left = librosa.stft(y[0], n_fft=NFFT, hop_length=HOPS)
        S_right = librosa.stft(y[1], n_fft=NFFT, hop_length=HOPS)
        return S_left, S_right
    
    if (mode == "mel"):
        mel_left = librosa.feature.melspectrogram(y=y[0], n_fft=NFFT, hop_length=HOPS, n_mels=512)
        mel_right = librosa.feature.melspectrogram(y=y[1], n_fft=NFFT, hop_length=HOPS, n_mels=512)
        return mel_left, mel_right
    

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

    if y is None or y.ndim != 2 or y.shape[1] == 0:
        raise ValueError("Input has to be a 2D non-empty array. Check again.")

    rms = librosa.feature.rms(y=librosa.to_mono(y[0]))[0]
    rms_left = librosa.feature.rms(y=y[0])[0]
    rms_right = librosa.feature.rms(y=y[1])[0]

    scaled_rms = asp_scaling(rms.reshape(-1, 1))
    scaled_left = asp_scaling(rms_left.reshape(-1, 1))
    scaled_right = asp_scaling(rms_right.reshape(-1, 1))

    rms_mean = np.rint(np.mean(scaled_rms))
    rms_left_mean = np.rint(np.mean(scaled_left))
    rms_right_mean = np.rint(np.mean(scaled_right))

    ## this indicated the energy difference between channels
    ## indicating panning
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
    The addition of 1e-10 tries to avoid, that the centroid leans to higher frequencies 
    for silent parts of the input signal
    """
    
    spec_cen_left = librosa.feature.spectral_centroid(y=y[0]+1e-10, sr=sr, n_fft=NFFT, hop_length=HOPS)[0]
    spec_cen_right = librosa.feature.spectral_centroid(y=y[1]+1e-10, sr=sr, n_fft=NFFT, hop_length=HOPS)[0]
    mean_spectral_centroid = np.rint(np.mean([spec_cen_left, spec_cen_right]))
    return mean_spectral_centroid, spec_cen_left, spec_cen_right

# scaling is odd (pure noise had values of 8 on a scale of 0..100)
def mean_spectral_flatness(
        y: Optional[np.ndarray]
        ) -> float:
    """
    Noisiness vs Tonalness
    
    This gives us hints about dry/wet settings and damping.
    The librosa return values are on a scale of [0,1] thus
    appropiate scaling for ASP. 
    - The closer the result to 0, the more tonal it is
    - The closer the result to 100, the more noisy it is
    """

    flatness = (librosa.feature.spectral_flatness(y=y, n_fft=NFFT, hop_length=HOPS))
    new = (flatness.reshape(-1,1)) * 100
    print(np.amin(new), np.amax(new), np.mean(new))
    mean_flatness = np.mean(new)
    return np.rint(mean_flatness)

def custom_flatness(y: np.ndarray):
    S = np.abs(librosa.stft(y=y, n_fft=NFFT, hop_length=HOPS))
    n_frames = S.shape[1]
    
    frame_flatness = np.zeros(n_frames)
    for i in range(n_frames):
        slice = S[:,i]
        geometric_mean = np.exp(np.mean(np.log(slice + 1e-10)))
        arithmetic_mean = np.mean(slice)

        res = geometric_mean / (arithmetic_mean + 1e-10)
        frame_flatness[i] = res

    mean = np.mean(frame_flatness) * 100

    return np.rint(mean)


#TODO: Check this method again (maybe devide by 2 since we take spread of both channels and calc the mean, thus having spread in both directions mixed in one value)
def spectral_spread(
        S_l: Optional[np.ndarray],
        S_r: Optional[np.ndarray],
        sr: float,
        centroid_left: np.ndarray,
        centroid_right: np.ndarray
        ) -> int:
    """
    Instantaneous bandwidth

    Hints at timbre by describing how stationary the sound is.
    Used for getting a better spectral resolution of the input.
    Spread is given in Hz, that means:
    - ~[0,500]: concentrated frequency content
    - ~[500-2000]: slight timbre variations
    - ~[>2000]: volatile timbre (quite possibly noisy and harsh)
    """

    freqs = librosa.fft_frequencies(sr=sr, n_fft=NFFT)
    S_left_mag = np.abs(S_l)
    S_right_mag = np.abs(S_r)
    S_left_norm = S_left_mag / (np.sum(S_left_mag, axis=0, keepdims=True) + 1e-10)
    S_right_norm = S_right_mag / (np.sum(S_right_mag, axis=0, keepdims=True) + 1e-10)

    spread_left = np.sqrt(np.sum(((freqs.reshape(-1,1) - centroid_left)**2) * S_left_norm, axis=0)/np.sum(S_left_norm))
    spread_right = np.sqrt(np.sum(((freqs.reshape(-1,1) - centroid_right)**2) * S_right_norm, axis=0)/np.sum(S_right_norm))
    return int(np.mean([np.mean(spread_left), np.mean(spread_right)]))

def compute_spectral_rolloff(
        y: Optional[np.ndarray],
        sr: float
        ) -> Tuple[np.ndarray[np.float32],np.ndarray[np.float32]]:
    """
    Indication about audio bandwith in low/high freqs

    Might be redundant concidering we use spectral centroid
    and spectral flatness.    
    """
    rolloff_left = librosa.feature.spectral_rolloff(y=y[0], sr=sr)[0]
    rolloff_right = librosa.feature.spectral_rolloff(y=y[1], sr=sr)[0]
    return rolloff_left, rolloff_right


def mid_side(y: Optional[np.ndarray]) ->Tuple[np.ndarray[np.float32],np.ndarray[np.float32]]:
    """Mid/side analysis for stereo spread parameter.
    Compared to rms_channel_balance, this indicates stereo/mono.
    """

    if len(y[0]) != len(y[1]):
        raise ValueError("Left and Right channel need to be of equal length")

    sum = (y[0] + y[1]) / SQRT_2
    diff = (y[0] - y[1]) / SQRT_2
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