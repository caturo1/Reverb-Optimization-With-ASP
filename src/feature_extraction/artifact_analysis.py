import numpy as np
from scipy import signal
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
    """
    Scaling of dB related values according to bit-depth of the signal

    Parameters:
        value: Input dB value
    
    Returns:
        out: Integer scaled value on a scale of [0,100]
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
def clipping_analyzer(y: np.ndarray):
    """
    Detection of clipping in one channel of the signal based on 
    "Detection of Clipped Fragments in Speech Signals" by Sergei Aleinik, Yuri Matveev
    
    Remark: It might be possible to detect clipping via fetching the maximal amplitude
    for ASP solving, but I thought a single clip isn't necessarily bad or a reason to solve again,
    but rather a certain clipping severity is.
    
    Parameters:
        y: One channel of stereo audio on [-1,1]

    Returns:
    Parameter indicating severity of clipping
    """

    # calculate an appropriate number of bins according to the input signal
    sturgers = np.log2(1 - (-1)) + 1

    num_bins = max(301, sturgers)

    # compute histogram
    amp_hist, _ = np.histogram(y, bins=num_bins)

    # initialize variables
    r_ptr = np.nonzero(amp_hist)[0][-1]
    l_ptr = np.nonzero(amp_hist)[0][0]
    current_r = amp_hist[r_ptr]
    current_l = amp_hist[l_ptr]
    d_l = 0
    d_r = 0
    d_max = 0

    denom = r_ptr - l_ptr

    # iterate over histogram bins
    while (r_ptr > l_ptr):

        if (l_ptr >= len(amp_hist) or r_ptr < 0):
            break

        r_ptr -= 1
        l_ptr += 1        
        y_r = amp_hist[r_ptr]
        y_l = amp_hist[l_ptr]

        # reset distance if a higher bin is found
        if (y_r > current_r):
            current_r = y_r
            d_r = 0
        # otherwise increment the distance
        else: d_r += 1
        
        # same procedure for the left side
        if (y_l > current_l):
            current_l = y_l
            d_l = 0
        else : d_l += 1

        # update fused distance value
        d_max = max(d_max, d_l, d_r)
    
    # calculate final clipping score
    r_cl = 2*d_max / denom if denom > 0 else 0

    return r_cl


def muddiness_analyzer(
                    y_proc: np.ndarray,
                    y_org: np.ndarray,
                    filter_bank_num_low: list,
                    filter_bank_num_mid: list
                    ):
    """
    Apply the previously generated gammatone filterbank to the final audio.
    Calculate the energy in distinct regions and perform differential analysis.
    This approximates perceptual changes in low/mid frequencies in order to detect a perceptual surge 
    of low energy that dangers the masking of mid/high content.

    Parameters:
        y_proc: Reverberated audio signal
        y_org: Unreverberated audio signal
        filter_bank_num_low: Transfer function coefficients for the gammatone filters of low end central frequencies
        filter_bank_num_mid: Transfer function coefficients for the gammatone filters of mid central frequencies

    Returns:
        mud_score: Score, that approximates perceptual changes
    """
    
    # Processing the mono signal is enough
    y_mono_proc = (y_proc[0] + y_proc[1]) / 2.
    y_mono_org = (y_org[0] + y_org[1]) / 2.

    # run the input through the filter
    bass_filtered_org = np.zeros_like(y_mono_org)
    bass_filtered_proc = np.zeros_like(y_mono_proc)
    mid_filtered_org = np.zeros_like(y_mono_org)
    mid_filtered_proc = np.zeros_like(y_mono_proc)

    for filter_coef in filter_bank_num_low:
        bass_filtered_org += signal.lfilter(filter_coef[0], filter_coef[1], y_mono_org)
        bass_filtered_proc += signal.lfilter(filter_coef[0], filter_coef[1], y_mono_proc)

    for filter_coef in filter_bank_num_mid:
        mid_filtered_org += signal.lfilter(filter_coef[0], filter_coef[1], y_mono_org)
        mid_filtered_proc += signal.lfilter(filter_coef[0], filter_coef[1], y_mono_proc)

    eps = 1e-10
    to_dB = lambda p : 10 * np.log10(p)
    # calculate and normalize values such as mean rms for comparative analysis
    low_energy_org = to_dB(np.mean(librosa.feature.rms(y=bass_filtered_org, frame_length=NFFT, hop_length=HOPS)))
    low_energy_proc = to_dB(np.mean(librosa.feature.rms(y=bass_filtered_proc, frame_length=NFFT, hop_length=HOPS)))
    mid_energy_org = to_dB(np.mean(librosa.feature.rms(y=mid_filtered_org, frame_length=NFFT, hop_length=HOPS)))
    mid_energy_proc = to_dB(np.mean(librosa.feature.rms(y=mid_filtered_proc, frame_length=NFFT, hop_length=HOPS)))
    
    total_org = low_energy_org + mid_energy_org + eps
    total_proc = low_energy_proc + mid_energy_proc + eps
    norm_org_low = low_energy_org / total_org
    norm_org_mid = mid_energy_org / total_org
    norm_proc_low = low_energy_proc / total_proc
    norm_proc_mid = mid_energy_proc / total_proc

    # compare energy distribution
    b2m_org = norm_org_low / norm_org_mid
    b2m_proc = norm_proc_low / norm_proc_mid

    mud_score = b2m_org - b2m_proc
    
    return mud_score

def muddiness_analyzation(mel_S: np.ndarray, mel_org: np.ndarray):
    """
    Approximation of perceived muddiness by using mel-Spektrograms. 
    To be compared to features of the original audio.
    
    This is a much more efficient, but not as mature approach.

    Parameters:
        mel_S: Mel Spectrogram of one channel ofthe processed audio
        mel_org: Mel Spectrogram of one channel ofthe original audio
    
    Returns:
        bass_to_mid_ratio: For 16-bit audio roughly [0,96]
    """

    mel_spec = librosa.power_to_db(mel_S)
    mel_spec_org = librosa.power_to_db(mel_org)
    
    n_bins = mel_spec.shape[1]
    mel_freqs = librosa.mel_frequencies(n_mels=n_bins,fmin=20, fmax=20_000)
    mel_concentrated = np.mean(mel_spec, axis=0)
    mel_concentrated_org = np.mean(mel_spec_org, axis=0)

    scores_org = {}
    scores_proc = {}
    for key in FREQ_BANDS:
        scores_org[key] = np.mean(mel_concentrated[FREQ_BANDS[key]["mask"](mel_freqs)])
        scores_proc[key] = np.mean(mel_concentrated_org[FREQ_BANDS[key]["mask"](mel_freqs)])

    # and how much bass/mids are make up the energy
    bass_to_mid_ratio_org = scores_org["bass"] - scores_org["mid"]
    bass_to_mid_ratio_proc = scores_proc["bass"] - scores_proc["mid"]
    
    b2m_channel = bass_to_mid_ratio_proc - bass_to_mid_ratio_org

    return b2m_channel

# implement with FFT?
def cross_correlation(y: Optional[np.ndarray]) -> float:
    """
    Calculation of cross correlation based on StereoProcessing paper by RS-MET.

    Parameters:
        y: 2D array with y[0] as left and y[1] as right channel
    
    Returns:
        correlation: In range of [-1,1]
    """

    if (y is None or y.ndim != 2 or y.shape[1] == 0):
        raise ValueError(f"Input has to be 2D and non-empty. Check again.")

    numerator = np.mean(y[0] * y[1])
    denom = np.sqrt((np.mean(y[0] ** 2) + np.mean(y[1] ** 2)) + 1e-10)
    c = (numerator / denom)

    return c


def get_frame_peak_density_spacing(mel_dB: np.ndarray) ->Tuple[np.ndarray, np.ndarray]:
    """
    Determine the peak density in each frame of out input spectrogram
    and the spacing of the peaks in every frame

    Parameters:
        mel_dB: Magnitude to dB converted spectrogram of the input audio

    Returns:
        peak_tracking: Matrix of size n_freqs x n_frames with count in respective position on [0,n_frames]
    """
    n_freqs, n_frames = mel_dB.shape
    freqs = librosa.mel_frequencies(n_mels=n_freqs, fmin=20, fmax=20_000)

    peak_tracking = np.zeros((n_freqs, n_frames))    

    for frame_idx in range(n_frames):
        peaks = librosa.util.peak_pick(
            mel_dB[:,frame_idx], 
            pre_max=3,
            post_max=3,
            pre_avg=10,
            post_avg=5,
            delta=0.4,
            wait=10)
        
        peak_tracking[[peaks], frame_idx] = 1
    
    return peak_tracking


def ringing(mel_proc: np.ndarray, mel_org: np.ndarray) -> int:
    """
    Sound event (ringing) analyzer. In some way this correlates with the "coloredness" of a reverb
    as it is defined as in "that their outputs impose some conspicuous audible 
    resonances upon the input signal." [Effect Design* 1: Reverberator and Other Filters JON DATTORR]
    
    Trying to detect ringing by analyzing the occurence of peaks in similar frequencies
    over consecutive frames. (This is somewhat a detection of standing waves/modals)

    We define surrounding areas/windows, for relevance and return the maximum
    for differential analysis.

    Parameters:
        S: STFT of processed audio

    Returns:
        differential_score: Severeness of ringing on [-2288,2288]
            in case every considered freq in every considered frame clips
    """


    peak_tracking_org = get_frame_peak_density_spacing(mel_org)
    peak_tracking_proc = get_frame_peak_density_spacing(mel_proc)

    ringing_org = compute_ringing_score(peak_tracking_org)
    ringing_proc = compute_ringing_score(peak_tracking_proc)

    differential_score = ringing_proc - ringing_org

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