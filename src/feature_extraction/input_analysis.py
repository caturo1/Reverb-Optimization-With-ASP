import librosa
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional

# store makros
RATE = 44_100
NFFT = 2048*2
HOPS = 512*2
SQRT_2 = np.sqrt(2)
# different frequency bands to calculate the gammatone filterbank
# please note: the frequency bands don't represent what's actually understood as bass/mid/high
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

def power_to_dB(value):
    """
    Convert power (rms) to dB on dBFS.
    Used for inputs on [0,1] (rms) and results in possible range of [-100,0] for eps.

    Parameters:
        value: power value to convert
    
    Returns:
        Decibel value
    """

    eps = 1e-10
    if (value >= 1e-10):
        return 10 * np.log10(value)
    else:
        return 10 * np.log10(eps)

def load_audio(file: str) -> Tuple[np.ndarray[np.float32],int]:
    """Load audio input as a stereo file
    
    Parameters:
        file: Path to the audio file
    Returns:
        y: The audio signal as a stereo signal
        sr: The sample rate of the audio file
    """
    y, sr = librosa.load(path=file, sr=RATE, mono=False)
    return to_stereo(y), sr
    
def to_stereo(
        y: np.ndarray[np.float32]
        ) -> np.ndarray[np.float32]:
    """
    Create stereo signal by duplicating the mono signal if necessary
    
    Parameters:
        y: The input audio signal, as a 2D numpy array.
    """
    if y is None:
        raise ValueError("Input signal cannot be None")

    # check if audio already is stereo
    mono = y.ndim == 1 or y.shape[0] == 1
    if not mono:
        return y
    
    # if mono, duplicate the signal to create stereo and stack channels vertically
    return np.vstack((y,y))


def compute_STFT(
        y: np.ndarray, 
        mode: str,
        ) -> Tuple[np.ndarray[np.float32],np.ndarray[np.float32]]:
    """
    Computation of specific spectrograms for both stereo channels.
    The function computes either a regular STFT or a mel spectrogram
    depending on the mode specified.

    Parameters:
        y: The input audio signal
        mode: The type of STFT to compute, either "regular" or "mel"
    """
    if (mode == "regular"):
        S_left = librosa.stft(y[0], n_fft=NFFT, hop_length=HOPS)
        S_right = librosa.stft(y[1], n_fft=NFFT, hop_length=HOPS)
        return S_left, S_right
    
    if (mode == "mel"):
        mel_left = librosa.feature.melspectrogram(y=y[0], n_fft=NFFT, hop_length=HOPS, n_mels=512)
        mel_right = librosa.feature.melspectrogram(y=y[1], n_fft=NFFT, hop_length=HOPS, n_mels=512)
        return mel_left, mel_right
    

def asp_scaling(arr) -> np.ndarray:
    """
    Scaler, that scales an array to a range of [0,100]
    for ASP reasoning. 

    Parameters:
        arr: The input array to be scaled
    """
    scaler = MinMaxScaler(feature_range=(0,100))
    return scaler.fit_transform(arr)

def dB_to_ASP(scalar):
    """
    Shifts dB values to a range of [0,100] while keeping the logarithmic dB scale

    Parameters:
        scalar: Value to be transformed

    Returns:
        Scaled value
    """

    return scalar + 100


def rms_features(
        y: np.ndarray
        ) -> Tuple[np.ndarray[np.float32], int, int, int]:
    """
    Root mean square value calculation of stereo channels.
    
    A central features that provides a measure for the energy contained
    in the signal. The results are transformed to an ASP-specific dB range for easy reasoning.
    
    Parameters:
        y: The input audio signal
    """

    # check if input is stereo (which it should as we transform every input to stereo)
    if y is None or y.ndim != 2 or y.shape[1] == 0:
        raise ValueError("Input has to be a 2D non-empty array. Check again.")

    # compute RMS for both channels
    rms = librosa.feature.rms(y=librosa.to_mono(y))[0]
    rms_left = librosa.feature.rms(y=y[0])[0]
    rms_right = librosa.feature.rms(y=y[1])[0]

    # calculate the means and transform them to our ASP-specific dB range
    rms_mean = dB_to_ASP((power_to_dB(np.mean(rms))))
    rms_left_mean = dB_to_ASP(power_to_dB(np.mean(rms_left)))
    rms_right_mean = dB_to_ASP(power_to_dB(np.mean(rms_right)))

    # calculate the channel balance as a simplification of actualy panning
    rms_channel_balance = np.abs(rms_left_mean - rms_right_mean)
    
    return rms, rms_mean, rms_channel_balance


def compute_dynamic_rms(rms: np.ndarray[np.float32]) -> int:
    """
    Dynamic range calculation based on rms features

    Calculation is done with percentiles to avoid outliers (ie. clips)
    to distort the dynamic range result.

    Parameters:
        rms: The input array of RMS values
    """

    if len(rms) == 0:
        return 0
    p97 = np.percentile(a=rms, q=97)
    p3 = np.percentile(a=rms, q=3)
    
    # instead of ratio, return difference in dB
    dr_dB = np.abs(power_to_dB(p97) - power_to_dB(p3))

    return dr_dB

def compute_dyn_range(y: np.ndarray[np.float32]) -> float:
    """
    Traditional dynamic range computation

    Parameters:
        y: The input audio signal

    Returns:
        Dynamic range in dB
    """

    # get absolute values of the signal
    flattened_y = np.abs(np.ravel(y))
    # compute min & max values while avoiding silent parts
    max_val = np.max(flattened_y)
    min_val = np.min(flattened_y[flattened_y > 1e-6])
    
    # check if signal is silent
    if max_val < 1e-6 or min_val < 1e-6: 
        return 0.0
        
    # compute dynamic range in dB
    dyn_r = 20 * np.log10(max_val / min_val)
    return dyn_r



def mean_spectral_centroid(
        S_l: Optional[np.ndarray],
        S_r: Optional[np.ndarray],
        sr: float
        ) -> Tuple[float,float,float]:
    """
    Calculates Center of gravity of spectral energy

    Parameters:
        S_l: The left channel spectrogram
        S_r: The right channel spectrogram
        sr: The sample rate of the audio file
    
    Returns:
        mean_spectral_centroid: The mean spectral centroid of both channels
        spec_cen_left: The spectral centroid of the left channel
        spec_cen_right: The spectral centroid of the right channel
    """
    
    spec_cen_left = librosa.feature.spectral_centroid(S=np.abs(S_l), sr=sr, n_fft=NFFT, hop_length=HOPS)[0]
    spec_cen_right = librosa.feature.spectral_centroid(S=np.abs(S_r), sr=sr, n_fft=NFFT, hop_length=HOPS)[0]
    mean_spectral_centroid = np.rint(np.mean([np.mean(spec_cen_left), np.mean(spec_cen_right)]))
    return mean_spectral_centroid, spec_cen_left, spec_cen_right

def custom_flatness(S: np.ndarray) -> int:
    """
    Calculation of spectral flatness.
    This measures noisiness vs tonalness.
    
    - The closer the result to 0, the more tonal it is
    - The closer the result to 1, the more noisy it is

    Parameters:
        S: The input spectrogram

    Returns:
        mean: The mean spectral flatness of the input
    """
    n_frames = S.shape[1]
    
    # calculate the geometric mean and arithmetic mean for each frame
    # and compute the flatness value
    frame_flatness = np.zeros(n_frames)
    for i in range(n_frames):
        slice = S[:,i]
        geometric_mean = np.exp(np.mean(np.log(slice + 1e-10)))
        arithmetic_mean = np.mean(slice)

        res = geometric_mean / (arithmetic_mean + 1e-10)
        frame_flatness[i] = res

    # scale the mean of [0,1] to [0,100] for ASP
    mean = np.mean(frame_flatness) * 100

    return np.rint(mean)


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

    Parameters:
        S_l: The left channel spectrogram
        S_r: The right channel spectrogram
        sr: The sample rate of the audio file
        centroid_left: The spectral centroid of the left channel
        centroid_right: The spectral centroid of the right channel
    
    Returns:
        mean: The mean spectral spread of both channels
    """

    # calculate the spectral centroid for both channels
    freqs = librosa.fft_frequencies(sr=sr, n_fft=NFFT)
    S_left_mag = np.abs(S_l)
    S_right_mag = np.abs(S_r)

    # normalize the spectral magnitudes
    S_left_norm = S_left_mag / (np.sum(S_left_mag, axis=0, keepdims=True) + 1e-10)
    S_right_norm = S_right_mag / (np.sum(S_right_mag, axis=0, keepdims=True) + 1e-10)

    # calculate the spread for both channels
    # the spread is calculated as the standard deviation of the spectral centroid
    spread_left = np.sqrt(np.sum(((freqs.reshape(-1,1) - centroid_left)**2) * S_left_norm, axis=0)/np.sum(S_left_norm))
    spread_right = np.sqrt(np.sum(((freqs.reshape(-1,1) - centroid_right)**2) * S_right_norm, axis=0)/np.sum(S_right_norm))
    
    return int(np.mean([np.mean(spread_left), np.mean(spread_right)]))

def generate_gammatone_filterbank(f_inf: int = 100,
                                  f_sup: int = 4000,
                                  n_bands: int = 25):
    
    """
    Create a filterbank with linearly spaced frequencies on the ERB scale.
    Inspired by matlab's implementation, with scipy's gammatone filters.

    Parameters:
        f_inf: Lower end of the frequency range in Hz
        f_sup: Upper end of the frequency range in Hz
        n_bands: Number of intermediate values, default = 35 as this is appropriate for a range of 0-4_000

    Returns: 
        filter_bank_num_low: The filter bank for low frequencies
        filter_bank_num_mid: The filter bank for mid frequencies
    """
        
    # define basic constants and transformations
    a = (1000 * np.log(10)) / (24.7 * 4.37)
    erb_2_hz = lambda x: (10 ** (x / a) - 1) / 0.00437
    hz_2_erb = lambda x: 21.4 * np.log10(0.00437 * x + 1)
    
    # calculate ERB-numbers for the specified range
    num_erbs_low = hz_2_erb(f_inf)
    num_erbs_high = hz_2_erb(f_sup)

    # create linspace array on the ERB-scale
    erb_num_array = np.linspace(start=num_erbs_low, stop=num_erbs_high, num=n_bands)

    # relate frequencies to ERBs
    center_freq = erb_2_hz(erb_num_array)
    # build filter banks seperately for different low & mid frequencies
    filter_bank_num_low = []
    filter_bank_num_mid = []
    for cf in center_freq:
        # Bandwidth of each filter is adjusted according to the ERB-formula in terms of center frequency.
        if (FREQ_BANDS["bass"]["mask"](cf)):
            b = signal.gammatone(freq=cf, ftype='iir', fs= RATE/2)
            filter_bank_num_low.append(b)
        if (FREQ_BANDS["mid"]["mask"](cf)):
            b = signal.gammatone(freq=cf, ftype='iir', fs= RATE/2)
            filter_bank_num_mid.append(b)
    
    return filter_bank_num_low, filter_bank_num_mid



def mid_side(y: Optional[np.ndarray]) ->Tuple[np.ndarray[np.float32],np.ndarray[np.float32]]:
    """
    Mid/side analysis for stereo spread parameter.
    Compared to rms_channel_balance, this indicates stereo/mono.

    Parameters:
        y: The input stereo audio signal
    
    Returns:
        mid: The mid value of the stereo signal
        side: The side value of the stereo signal
    """

    if len(y[0]) != len(y[1]):
        raise ValueError("Left and Right channel need to be of equal length")

    sum = (y[0] + y[1]) / SQRT_2
    diff = (y[0] - y[1]) / SQRT_2
    # scale values to [0,100] for ASP reasoning while keeping the relative differences
    scaled_sum = asp_scaling(sum.reshape(-1,1))
    scaled_diff = asp_scaling(diff.reshape(-1,1))
    mid = np.rint(np.mean(scaled_sum))
    side = np.rint(np.mean(scaled_diff))
    
    return mid, side