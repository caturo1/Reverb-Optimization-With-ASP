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

"""
There are still inconsistencies as to handling differential analysis in
the function or in the ArtifactFeatures class. Soon to be resolved.
"""

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
    
    Remark: It might be possible to detect clipping via fetching the maximal amplitude
    for ASP solving, but I thought a single clip isn't necessarily bad or a reason to solve again,
    but rather a certain clipping severity is.
    
    Parameter:
    - y: One channel of stereo audio on [-1,1]

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

    return r_cl


# Check here again
def muddiness_analyzation(mel_S: np.ndarray):
    """
    Approximation of perceived muddiness. To be compared to features of the original audio.
    We could also try to calculate the spectral centroid and compare the shift in the centroid between original and processed audio.
    We could also calculate the spectral rolloff to see if there are changes 85 (or so) percentile energy distributions in lower freqs

    Not sure if melspectrum or perceptually weighted STFT is better. 
    The latter has a better frequency resolution.

    Parameters:
    ----------
        mel_S: Mel Spectrogram of one channel
    
    Return:
    -------
        bass_to_mid_ratio: For 16-bit audio roughly [0,96]
    """

    mel_spec = librosa.power_to_db(mel_S)
    
    n_bins = mel_spec.shape[1]
    mel_freqs = librosa.mel_frequencies(n_mels=n_bins,fmin=20, fmax=20_000)
    mel_concentrated = np.mean(mel_spec, axis=0)

    scores = {}
    for key in FREQ_BANDS:
        scores[key] = np.mean(mel_concentrated[FREQ_BANDS[key]["mask"](mel_freqs)])

    # and how much bass/mids are make up the energy
    bass_to_mid_ratio = scores["bass"] - scores["mid"]
    
    return bass_to_mid_ratio

# implement with FFT?
def cross_correlation(y: Optional[np.ndarray]) -> float:
    """
    Calculation of cross correlation based on StereoProcessing paper.

    Parameters:
    y: 2D array with y[0] as left and y[1] as right channel
    
    Return:
    correlation: In range of [-1,1]
    """

    if (y is None or y.ndim != 2 or y.shape[1] == 0):
        raise ValueError(f"Input has to be 2D and non-empty. Check again.")

    numerator = np.mean(y[0] * y[1])
    denom = np.sqrt((np.mean(y[0] ** 2) + np.mean(y[1] ** 2)) + 1e-10)
    ## changed resulting values
    c = (numerator / denom)

    return c


def get_frame_peak_density_spacing(mel_dB: np.ndarray) ->Tuple[np.ndarray, np.ndarray]:
    """
    Determine the peak density in each frame of out input spectrogram
    and the spacing of the peaks in every frame

    Parameters:
    -----------
        mel_dB: Magnitude to dB converted spectrogram of the input audio

    Return:
    -------
        density: density measurement in relation to whole audio on [0,1]
        peak_tracking: Matrix of size n_freqs x n_frames with count in respective position on [0,n_frames]
        spacing: Array containing the frequency-dependent space between peaks in Hz on [0,19_980]
    """
    n_freqs, n_frames = mel_dB.shape
    freqs = librosa.mel_frequencies(n_mels=n_freqs, fmin=20, fmax=20_000)

    peak_tracking = np.zeros((n_freqs, n_frames))
    density = np.zeros(shape=n_frames, dtype=float)
    spacing = np.full(shape=n_frames * (n_freqs-1), fill_value=np.nan)
    
    index = 0
    for frame_idx in range(n_frames):
        peaks = librosa.util.peak_pick(
            mel_dB[:,frame_idx], 
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=3,
            delta=0.35,
            wait=10)
        
        # this  gives me just how many peaks I gave per frame in relation to the whole audio
        # I could expand this to the different frequency bands
        density[frame_idx] = len(peaks) / n_frames
        
        peak_tracking[[peaks], frame_idx] = 1

        if len(peaks) > 1:
            diffs = np.diff(freqs[peaks])
            upper = len(diffs)
            spacing[index : index + upper] = diffs
            index += upper
    spacing = spacing[~np.isnan(spacing)]

    return density, peak_tracking, spacing


def spectral_density(mel_org: np.ndarray, mel_proc: np.ndarray) -> Tuple[int, int]:
    """
    Rough measurement of spectral density. 
    The difference between the input and processed audio gives us a measurement about the extent of reverb we can apply
    
    Parameters:
    -----------
        S_org: STFT of the original audio
        S_proc: STFT of the processed audio

    Returns:
    --------
        density_stability: Desc in code; on scale of [0,1]
        peak_density_difference: Accumulation of peaks on [-n_frames,n_frames]

    """
    n_frames = mel_dB_org.shape[1]

    mel_dB_org = librosa.amplitude_to_db(mel_org)
    mel_dB_proc = librosa.power_to_db(mel_proc)

    spectral_density_org, _, _ = get_frame_peak_density_spacing(mel_dB=mel_dB_org)
    spectral_density_proc, _, _ = get_frame_peak_density_spacing(mel_dB=mel_dB_proc)
    
    spectral_density_org = np.where(spectral_density_org != 0, spectral_density_org, 1e-10)
    spectral_density_proc = np.where(spectral_density_proc != 0, spectral_density_proc, 1e-10)
    
    # measurement of how much the density remains stable between processing
    ## density_ratio contains a deensity value per frame
    ## density_ratio = 1 if both are the same
    ## density_ratio > 1 if we introduced more peaks
    ## density_ratio < 1 if we resolved peaks
    ## density_deriv contains only relevant frames
    ## stability measures the extend for relevant (aka introduced) peaks
    density_ratio = spectral_density_proc / spectral_density_org
    cond = (density_ratio != 1) & ~np.isnan(density_ratio)
    density_deriv = np.extract(cond, density_ratio)
    density_stability = (1 - len(density_deriv) / n_frames)
    
    # measure to which extend the peak density accumulated or disappeared
    ## combined contains the peak-count difference
    ## peak_density_difference is positive if we introduced peaks
    ## and relates it to the size of the audio
    ## becuase 20 new peaks in a 1 s audio are a lot in a 1 minute audio not 
    combined = spectral_density_proc[spectral_density_proc] - spectral_density_org[spectral_density_org]
    pos_derv = len(np.extract(combined > 0, combined))
    neg_derv = len(np.extract(combined < 0, combined))
    peak_density_difference = pos_derv - neg_derv
    
    return density_stability, peak_density_difference

def spectral_clustering(mel_org: np.ndarray, mel_proc: Optional[np.ndarray]):
    """
    Detect clustering of peaks to detect irritating resonances. 
    This is a differential analysis to see if we introduce new resonances.
    We compare it to the clustering of the original input in order to avoid 
    adjusting parameters for the wrong error source (false positives). 
    This is relevant for basically every reverb parameter.
    We also calculate the regularity of spacing between peaks to detect unnatural reverbration patterns (not anymore, that was stupid)

    Parameters:
    -----------
        S_org: STFT of the original audio
        S_proc: STFT of the processed audio

    Returns:
    --------
        final_score: Weighted score compined of closely clustered 
            and dangerously clustered peaks [-511,511]
    """

    mel_dB_org = librosa.power_to_db(mel_org)
    mel_dB_proc = librosa.power_to_db(mel_proc)

    density_org, _, spacing_org = get_frame_peak_density_spacing(mel_dB=mel_dB_org)
    density_proc,_, spacing_proc = get_frame_peak_density_spacing(mel_dB=mel_dB_proc)
    
    if (len(spacing_org) < 2 or len(spacing_proc) < 2):
        print("Reverbrated audio is too short to properly evaluate spacing parameters")
        return (0, 0, 0, 0)

    # peak clusters for mean density (shouldn't be too low nor too high and might tell us about resonances)
    o_cluster_score, o_resonance_score = compute_clustering(spacing_org, density_org)
    p_cluster_score, p_resonance_score = compute_clustering(spacing_proc, density_proc)

    clustering_diff = p_cluster_score - o_cluster_score
    resonance_diff = p_resonance_score - o_resonance_score

    final_score = (clustering_diff * 0.3 + resonance_diff * 0.7)
    return final_score

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
        return (0.0, 0.0)


    # density values in range of [0,1]
    # threshold have to be exaluated experimentally
    # both values however are for differential analysis anyway
    cluster_score = sum(spacing < 60)
    resonance_score = sum(spacing  < 30)

    return (cluster_score, resonance_score)


def ringing(mel_proc: np.ndarray, mel_org: np.ndarray) -> int:
    """
    Ringing frequency analyzer
    
    Trying to detect ringing by analyzing the occurence of peaks in similar frequencies
    over consecutive frames.
    We define surrounding areas/windows, for relevance and return the maximum
    for differential analysis.

    Parameters:
    -----------
        S: STFT of processed audio

    Return:
    -------
        differential_score: Severeness of ringing on [-2288,2288]
            in case every considered freq in every considered frame clips
    """


    _, peak_tracking_org, _= get_frame_peak_density_spacing(mel_org)
    _, peak_tracking_proc, _= get_frame_peak_density_spacing(mel_proc)

    ringing_org = compute_ringing_score(peak_tracking_org)
    ringing_proc = compute_ringing_score(peak_tracking_proc)

    differential_score = ringing_proc - ringing_org
    print(f"The differential score: {differential_score}")

    return differential_score

def compute_ringing_score(peak_tracking: np.ndarray) -> int:
    
    n_freqs = peak_tracking.shape[0]
    n_frames = peak_tracking.shape[1]
    # time interval of 500 ms
    time_eps = int(np.rint((RATE * 0.5) / HOPS))
    time_steps = int(np.rint(n_frames/time_eps*2))

    # frequency interval of 5% of frequency bins
    freq_eps = int(np.rint(n_freqs * 0.05))
    freq_steps = int(np.rint(n_freqs/freq_eps*2))

    # calculate the peak occurences in 2D window to check
    # how many closely spaced, time-related peaks are there
    ringing = np.zeros(shape=(freq_steps, time_steps))
    frame_idx = 0
    array_t_idx = 0
    while (frame_idx := frame_idx + time_eps) <= n_frames - time_eps:
        freq_idx = 0
        array_f_idx = 0
        while (freq_idx := freq_idx + freq_eps) <= n_freqs - freq_eps:
            ringing[array_f_idx, array_t_idx] = (np.sum(
                                            peak_tracking[freq_idx - freq_eps : freq_idx + freq_eps,
                                            frame_idx - time_eps : frame_idx + time_eps]))
            array_f_idx += 1
        array_t_idx += 1

    ringing_score = np.max(ringing) if ringing.size > 0 else 0

    return ringing_score