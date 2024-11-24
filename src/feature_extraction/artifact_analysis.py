import numpy as np
from typing import Optional, Tuple
import librosa
from AudioFeatures import AudioFeatures

RATE = 44_100
FREQ_BANDS = {
    "bass" : {
        "range" : (0,400),
        "mask" : lambda f : (f >= 20) & (f <= 400)
    },

    "mid" : {
        "range" : (401,4000),
        "mask" : lambda f : (f > 500) & (f <= 4000)
    },

    "high" : {
        "range" : (4001, 20_000),
        "mask" : lambda f : (f > 4000)
    }
}

# In general might have to pay attention to the frame length again
# and maybe use STFT and perceptual weighting instead of mel scale
# since this might cluster to many frequencies into one bin

# Check here again
def muddiness_analyzation(y_processed: np.ndarray, mel_S: Optional[np.ndarray], input: AudioFeatures):
    """
    Approximation of perceived muddiness. To be compared to features of the original audio.
    We could also try to calculate the spectral centroid and compare the shift in the centroid between original and processed audio.
    We could also calculate the spectral rolloff to see if there are changes 85 (or so) percentile energy distributions in lower freqs

    Not sure if melspectrum or perceptually weighted STFT is better. 
    The latter has a better frequency resolution.
    """

    mel_spec = librosa.amplitude_to_db(np.abs(mel_S))

    mel_freqs = librosa.mel_frequencies(n_mels=mel_spec.shape[0], sr=RATE)

    # get mean dB for each frequency band based on mel spec
    scores = {
        key: np.mean(mel_spec[:,FREQ_BANDS[key]["mask"](mel_freqs)]) for key in FREQ_BANDS
    }

    mean = np.mean(scores[key] for key in scores)

    bass_ratio = scores["bass"] - mean
    mid_ratio = scores["mid"] - mean
    bass_to_mid_ratio = scores["bass"] - scores["mid"]
    high_total = scores["high"] - sum(scores[key] for key in scores)

    # maybe adjust the thresholds or assign differenct weights
    overall_score = (
        (mid_ratio / -4.5) * 40 +
        (1 - bass_to_mid_ratio / 1.6) * 30 +
        (1 - high_total / -6.0) * 30
        ) * 100
    
    return bass_ratio, mid_ratio, bass_to_mid_ratio, overall_score


def cross_correlation(y: Optional[np.ndarray]) -> float:
    """
    Calculation of cross correlation based on StereoProcessing paper.

    Parameters:
    y: 2D array with y[0] as left and y[1] as right channel
    
    Return:
    correlation: Range between [0,200] for ASP guessing where
    - prev 0 -> 100
    - prev 1 -> 200
    - prev -1 -> 0
    """

    if (y is None or y.ndim != 2 or y.shape[1] == 0):
        raise ValueError(f"Input has to be 2D and non-empty. Check again.")

    numerator = np.mean(y[0] * y[1])
    denom = np.sqrt((np.mean(y[0] ** 2) + np.mean(y[1] ** 2)) + 1e-10)
    c = ((numerator / denom) + 1) * 100

    return c

# should normalize before peak detection
def get_frame_peak_density_spacing(mag_dB):
    """
    Determine the peak density in each frame of out input spectrogram
    and the spacing of the peaks in every frame

    Parameters:
    Magnitude to dB converted spectrogram of the input audio
    """
    
    density = []
    spacing = []
    peak_array = []
    for frame in mag_dB.T:
        peaks = librosa.util.peak_pick(frame, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.5, wait=10)
        peak_array.append(peaks)

        if peaks > 1:
            peak_spacing = np.diff(peaks)
            density.append(len(frame) / len(peaks))
            spacing.append(peak_spacing)
    
    return np.array(density), np.array(spacing)


def spectral_density(S_org: np.ndarray, S_proc: np.ndarray) -> Tuple[int, int]:
    """
    Rough measurement of spectral density. 
    The difference between the input and processed audio gives us a measurement about the extent of reverb we can apply
    
    Parameters:
    S_org: STFT of the original audio
    S_proc: STFT of the processed audio
    """
    
    mag_org = np.abs(S_org)
    mag_dB_org = librosa.amplitude_to_db(mag_org)

    mag_proc = np.abs(S_proc)
    mag_dB_proc = librosa.amplitude_to_db(mag_proc)

    spectral_density_org, _ = get_frame_peak_density_spacing(mag_dB=mag_dB_org)
    spectral_density_proc, _ = get_frame_peak_density_spacing(mag_dB=mag_dB_proc)

    density_ratio = spectral_density_proc / spectral_density_org
    density_difference = spectral_density_proc - spectral_density_org

    return density_ratio, density_difference

def spectral_clustering(S_org: np.ndarray, S_proc: Optional[np.ndarray]):
    """
    Detect clustering of peaks to detect irritating resonances. 
    We compare it to the clustering of the original input in order to avoid 
    adjusting parameters for the wrong error source. 
    This is relevant for basically every reverb parameter.

    Parameters:
    S_org: STFT of the original audio
    S_proc: STFT of the processed audio
    """
    mag_org = np.abs(S_org)
    mag_dB_org = librosa.amplitude_to_db(mag_org)

    mag_proc = np.abs(S_proc)
    mag_dB_proc = librosa.amplitude_to_db(mag_proc)

    density_org, spacing_org = get_frame_peak_density_spacing(mag_dB_org)
    density_proc, spacing_proc = get_frame_peak_density_spacing(mag_dB_proc)

    if (len(spacing_org) < 2 or len(spacing_proc) < 2):
        return 0
    
    # the regularity of the spacing might indicate ringing and naturalness of reverb (the more irregular the better)
    o_spacing_regularity = 1 - (np.std(spacing_org) / np.mean(spacing_org))
    p_spacing_regularity = 1 - (np.std(spacing_proc) / np.mean(spacing_proc))

    # peak clusters for mean density (shouldn't be too low nor too high and might tell us about resonances)
    o_cluster_score = sum(spacing_org < np.mean(spacing_org) * 0.5) / np.mean(density_org)
    o_resonance_score = sum(spacing_org < np.mean(spacing_org) * 0.35)

    p_cluster_score = sum(spacing_org < np.mean(spacing_proc) * 0.5) / np.mean(density_proc)
    p_resonance_score = sum(density_proc < np.mean(density_proc) * 0.35)

    spacing_regularity_diff = o_spacing_regularity - p_spacing_regularity
    clustering_diff = o_cluster_score - p_cluster_score
    resonance_diff = o_resonance_score - p_resonance_score

    return int(spacing_regularity_diff), int(clustering_diff), int(resonance_diff), int(p_resonance_score), int(p_cluster_score)

# have to handle normalization before peak detection
# I could apply perceptual weighting
def ringing(S: np.ndarray, sr, frame_len):
    """
    Ringing frequency analyzer
    
    Trying to detect ringing by analyzing the occurence of peaks in similar frequency over STFT frames.
    Might be problematic due to non-periodic, possibly random nature of input audio. 
    The perceptual weighting of the STFT gives us a good frequency resolution on a perceptual scale.

    Parameters:
    S: STFT of processed audio
    """
    
    mag_dB = np.abs(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_len)

    peak_tracking = np.zeros(len(freqs),mag_dB.shape[1])
    for frame_idx, frame in enumerate(mag_dB.T):
        peaks = librosa.util.peak_pick(frame, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.5, wait=10)
        
        peak_tracking[peaks, frame_idx] = 1

    lingering = sum(peak_tracking > np.mean(peak_tracking) + np.std(np.sum(peak_tracking, axis=1)))

    return lingering

