import numpy as np
from typing import Optional, Tuple
import librosa

RATE = 44_100
NFFT = 2048*2
HOPS = 512*2
BIT_DEPTH = 16
FREQ_BANDS = {
    "bass" : {
        "range" : (0,400),
        "mask" : lambda f : f <= 400
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

def db_scaling(value) -> int:
    """Scaling of dB related values according to bit-depth of the signal

    Args: 
        value: Input dB value
    
    Returns:
        Integer scaled value on a scale of [0,100]
    """
    min = -20*np.log10(2**BIT_DEPTH)
    max = 0
    eps = 1e-10
    nom = value - min
    denum = max - min + eps
    x_scaled = nom / denum

    return np.rint(max(100,x_scaled))

# In general might have to pay attention to the frame length again
# and maybe use STFT and perceptual weighting instead of mel scale
# since this might cluster to many frequencies into one bin
# also, most methods work for one channel so either expand them to stereo input or apply them to each channel

# TODO Implement clipping detector (like 3 successive amplitude peaks above threshold like -0.3 dB or so --audacity's method )
# Also possible to analyze a histogram and see if we have significant peaks at high levels
def clipping_analyzer(y: np.ndarray):
    """
    Detection of clipping in one channel of the signal based on 
    "Detection of Clipped Fragments in Speech Signals" by Sergei Aleinik, Yuri Matveev

    Parameter:
    - y: One channel of stereo audio

    Returns:
    Parameter indicating severity of clipping
    """

    sturgers = np.log2(1 - (-1)) + 1

    num_bins = max(301, sturgers)

    amp_hist, _ = np.histogram(y, bins=num_bins)
    
    r_ptr = np.nonzero(amp_hist)[0][-1]
    l_ptr = np.nonzero(amp_hist)[0][0]
    current_r = amp_hist[r_ptr]
    current_l = amp_hist[l_ptr]
    d_l = 0
    d_r = 0
    d_max = 0

    denom = r_ptr - l_ptr

    while (r_ptr > l_ptr):

        if (l_ptr >= len(amp_hist) or r_ptr < 0):
            break

        r_ptr -= 1
        l_ptr += 1        
        y_r = amp_hist[r_ptr]
        y_l = amp_hist[l_ptr]

        if (y_r > current_r):
            current_r = y_r
            d_r = 0
        else: d_r += 1
        
        if (y_l > current_l):
            current_l = y_l
            d_l = 0
        else : d_l += 1

        d_max = max(d_max, d_l, d_r)
    
    r_cl = 2*d_max / denom if denom > 0 else 0

    return int(min(1.0, r_cl)*100)


# Check here again
def muddiness_analyzation(mel_S: np.ndarray):
    """
    Approximation of perceived muddiness. To be compared to features of the original audio.
    We could also try to calculate the spectral centroid and compare the shift in the centroid between original and processed audio.
    We could also calculate the spectral rolloff to see if there are changes 85 (or so) percentile energy distributions in lower freqs

    Not sure if melspectrum or perceptually weighted STFT is better. 
    The latter has a better frequency resolution.
    """

    mel_spec = librosa.power_to_db(mel_S)
    
    n_bins = mel_spec.shape[1]
    mel_freqs = librosa.mel_frequencies(n_mels=n_bins)
    mel_concentrated = np.mean(mel_spec, axis=0)

    scores = {}
    for key in FREQ_BANDS:
        scores[key] = np.mean(mel_concentrated[FREQ_BANDS[key]["mask"](mel_freqs)])

    # and how much bass/mids are make up the energy
    bass_to_mid_ratio = np.abs(scores["bass"] - scores["mid"])
    
    return np.rint(bass_to_mid_ratio)

# implement with FFT?
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


def get_frame_peak_density_spacing(mag_dB: np.ndarray) ->Tuple[np.ndarray, np.ndarray]:
    """
    Determine the peak density in each frame of out input spectrogram
    and the spacing of the peaks in every frame

    Parameters:
    Magnitude to dB converted spectrogram of the input audio
    """
    n_freqs, n_frames = mag_dB.shape
    freqs = librosa.fft_frequencies(sr=RATE, n_fft=NFFT)
    
    # in the process of changing this method
    peak_tracking = np.zeros(n_frames, n_freqs)
    density = np.zeros(shape=n_frames, dtype=float)
    spacing = []

    
    for frame_idx in range(n_frames):

        peaks = librosa.util.peak_pick(
            mag_dB[:,frame_idx], 
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=3,
            delta=0.35,
            wait=10)
        density[frame_idx] = len(peaks) / n_freqs
        
        if len(peaks) > 1:
            peak_spacing = np.diff(peaks)
            spacing.append(peak_spacing)
        else:
            spacing.append(np.array([]))

    return density, spacing


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
    
    spectral_density_org = np.where(spectral_density_org != 0, spectral_density_org, 1e-10)
    spectral_density_proc = np.where(spectral_density_proc != 0, spectral_density_proc, 1e-10)
    
    # measurement of how much the density remains stable between processing
    density_ratio = spectral_density_proc / spectral_density_org
    cond = (density_ratio != 1) & ~np.isnan(density_ratio)
    density_deriv = np.extract(cond, density_ratio)
    density_stability = (1 - len(density_deriv) / mag_dB_org.shape[1]) * 100
    
    # measure to which extend the peak density accumulated or disappeared
    combined = spectral_density_proc - spectral_density_org
    pos_derv = len(np.extract(combined > 0, combined))
    neg_derv = len(np.extract(combined < 0, combined))
    peak_density_difference = ((pos_derv - neg_derv) / len(spectral_density_org))*100
    
    return density_stability, peak_density_difference

def spectral_clustering(mel_org: np.ndarray, mel_proc: Optional[np.ndarray]):
    """
    Detect clustering of peaks to detect irritating resonances. 
    We compare it to the clustering of the original input in order to avoid 
    adjusting parameters for the wrong error source (false positives). 
    This is relevant for basically every reverb parameter.
    We also calculate the regularity of spacing between peaks to detect unnatural reverbration patterns

    Parameters:
    S_org: STFT of the original audio
    S_proc: STFT of the processed audio
    """

    mag_dB_org = librosa.power_to_db(mel_org)
    mag_dB_proc = librosa.power_to_db(mel_proc)

    density_org, spacing_org = get_frame_peak_density_spacing(mag_dB_org)
    density_proc, spacing_proc = get_frame_peak_density_spacing(mag_dB_proc)
    
    if (len(spacing_org) < 2 or len(spacing_proc) < 2):
        print("Reverbrated audio is too short to properly evaluate spacing parameters")
        return (0, 0, 0, 0)

    # peak clusters for mean density (shouldn't be too low nor too high and might tell us about resonances)
    o_cluster_score, o_resonance_score = compute_clustering(spacing_org, density_org)
    p_cluster_score, p_resonance_score = compute_clustering(spacing_proc, density_proc)

    clustering_diff = o_cluster_score - p_cluster_score
    resonance_diff = o_resonance_score - p_resonance_score

    return int(clustering_diff)*100, int(resonance_diff)*100

def compute_clustering(spacing: np.ndarray, density: np.ndarray) -> float:
    """
    Calculates the extend of peaks, that cluster closely together
    and relates it to the overall peak density for comparability

    Parameters:
    -----------
    spacing: np.ndarray
        multi-dimensional array with the sample difference between peaks
    density: np.ndarray
        one-dimensional array containing the peak densities per frame

    Returns:
    --------
    cluster_score : float 
        Relation of number of close spacing to the density
    resonance_score : float
        Number of very close spectral peaks
    """
    if spacing.size == 0 or density.size == 0:
        return 0.0, 0.0

    mean = np.mean(spacing)
    cluster_score = sum(spacing[spacing < mean * 0.5] ) / np.mean(density)
    resonance_score = sum(spacing  < (mean * 0.35))

    return cluster_score, resonance_score


# have to handle normalization before peak detection
# I could apply perceptual weighting
def ringing(S: np.ndarray, sr):
    """
    Ringing frequency analyzer
    
    Trying to detect ringing by analyzing the occurence of peaks in similar frequency over STFT frames.
    Might be problematic due to non-periodic, possibly random nature of input audio. 
    The perceptual weighting of the STFT gives us a good frequency resolution on a perceptual scale.

    Parameters:
    S: STFT of processed audio
    """
    
    mag_dB = np.abs(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048*2)

    peak_tracking = np.zeros(len(freqs),mag_dB.shape[1])
    for frame_idx, frame in enumerate(mag_dB.T):
        peaks = librosa.util.peak_pick(frame, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.5, wait=10)
        
        peak_tracking[peaks, frame_idx] = 1

    lingering = sum(peak_tracking > np.mean(peak_tracking) + np.std(np.sum(peak_tracking, axis=1)))
    
    # maybe wrong scaling
    ringing_score = int((lingering/mag_dB.shape[1])*100)

    return ringing_score
