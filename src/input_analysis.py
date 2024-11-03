import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

CHUNK = 1024
CHANNELS = 2
RATE = 44_100

def normalize_signal(sig):
    rms = np.sqrt(np.mean(sig**2))
    return sig / rms if rms > 0 else sig

def check_sr(name):
    return librosa.get_samplerate(name)

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

def rms_to_db(rms):
    return np.abs(30 + 20 * np.log10(rms))

def load_audio(file) -> tuple:
    y, sr = librosa.load(path=file, sr=RATE, mono=False)
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

# window_size = n_fft size (default 2048)
def compute_rms(S):
    return librosa.feature.rms(S=S)

def compute_dynamic_rms(rms):
    # if stereo, concentrate array to 1D for computation
    if rms.size > 1:
        rms = np.concatenate(rms, axis=0)
    
    # so we don't run into division by 0
    rms = rms[rms > 0]
    #return rms ratio in dB
    return 20 * np.log10(np.max(rms) / np.min(rms))

# idk, not sure here
def compute_dynamic_snr(y):
    if y.ndim == 1:  # Mono signal
        m = y.mean()
        sd = y.std()
    elif y.ndim == 2:  # Stereo signal
        m = y.mean(axis=0) + 1e-20  # Mean for each channel
        sd = y.std(axis=0)  # Standard deviation for each channel
    else:
        raise ValueError("Input signal must be 1D (mono) or 2D (stereo).")
    
    # Ensure m / sd is a valid positive number
    ratio = m / sd
    if np.any(ratio <= 0):
        return 0  # or some other appropriate value or handling
    
    return 20 * np.log10(np.abs(ratio)).mean()  # Average SNR across channels