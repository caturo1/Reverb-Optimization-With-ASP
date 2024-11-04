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

def normalize_signal(sig):
    rms = np.sqrt(np.mean(sig**2))
    return sig / rms if rms > 0 else sig

def check_sr(name):
    return librosa.get_samplerate(name)

"""
def spectral_plot(S_db, y, centroid, flatness, contrast, sr):
    # Create plot
    fig, ax = plt.subplots(3, sharex=True, figsize=(12, 8))
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='hz', sr=sr, ax=ax[0])
    fig.colorbar(img, ax=[ax[0]], format="%+2.0f dB")
    
    # Plot centroid and flatness with time-aligned frames
    times = librosa.times_like(centroid, sr=sr)
    ax[0].plot(times, centroid, label='Spectral centroid', color='w', linewidth=2)
    ax[0].set(title='Spectrogram with Spectral Centroid, Flatness, and Contrast')
    ax[0].legend(loc='upper right')
    
    # Plot Contrast
    img2 = librosa.display.specshow(contrast, x_axis='time', ax=ax[1], sr=sr)
    fig.colorbar(img2, ax=[ax[1]])
    ax[1].set(ylabel='Frequency bands', title='Spectral contrast')

    # Plot Flatness
    times_flatness = librosa.times_like(flatness, sr=sr)
    librosa.display.waveshow(y, sr=sr, ax=ax[2], color='gray', alpha=0.5) 
    ax[2].plot(times_flatness, flatness, color='g', label='Spectral Flatness scaled by a factor of 100')
    ax[2].set(ylabel='Flatness', title='Spectral Flatness')
    ax[2].legend(loc='upper right')

    ax[2].set_ylim([0, 0.1])


    plt.tight_layout()
    plt.show()
"""

def rms_to_db(rms):
    return np.abs(30 + 20 * np.log10(rms))

def load_audio(file):
    # Convert Samples to dBFS
    """R_dB = -6
    R = 10^(R_dB/20)"""
    y, sr = librosa.load(path=file, sr=RATE, mono=True)
    return y, sr

"""
To compute the 
- Magnitude spectrum we use np.abs(S)
- Phase spectrum we use np.angle(S)
"""
def compute_STFT(y, sr):
    S = librosa.stft(y)
    return S

def mean_spectral_centroid(y, sr):
    return librosa.feature.spectral_centroid(y=y, sr=sr)[0]

def mean_spectral_flatness(y):
    return librosa.feature.spectral_flatness(y=y)[0]

def compute_spectral_rolloff(y, sr):
    return librosa.feature.spectral_rolloff(y=y, sr=sr)

def median_spectral_contrast(S, sr):
    return librosa.feature.spectral_contrast(S=np.abs(S), sr=sr)

def rms_to_dB(rms, eps=1e-20):
    # same as 20*log10(rms+eps)
    return librosa.power_to_db(rms**2)

# window_size = n_fft size (default 2048)
# not sure, if the scaling is done properly or of it distorts the feature perception
def rms_features(y) -> Tuple[npt.NDArray[np.float32], int]:
    scaler = MinMaxScaler(feature_range=(0,100))
    rms = librosa.feature.rms(y=y)
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
