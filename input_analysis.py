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

def main():
    filename = "./data/schreihals.wav"

    # Load and compute spectrogram
    y, sr = librosa.load(filename, sr=RATE)
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # Compute spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # Compute spectral flatness
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    scaled_flatness = flatness*100

    # Compute spectral roll-off
    s_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # Compute spectral contrast
    contrast = librosa.feature.spectral_contrast(S=np.abs(S), sr=sr)

    # Plot the spectrogram with features
    spectral_plot(S_db, y, centroid=centroid, flatness=scaled_flatness, contrast=contrast, sr=sr)

    # Compute RMS
    s_rms = librosa.feature.rms(S=S)
    dynamics_db_rms = rms_to_db(s_rms.max() - s_rms.min())

    print(f"The mean spectral flatness is {flatness.mean():.4f} with maximum rolloff at {s_rolloff.max():.2f}\n"
          f"and a root mean squared between {s_rms.min():.4f} and {s_rms.max():.4f} leading to a general rms dynamic range of {dynamics_db_rms:.2f}\n"
          f"and a mean spectral centroid at {centroid.mean():.2f} Hz and {flatness}")

if __name__ == "__main__":
    main()
