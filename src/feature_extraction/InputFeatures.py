import os
import numpy as np
from src.application.AspHandler import AspHandler
from . import input_analysis as ia
from typing import Optional

# important: For all spectral features, instead of passing y, pass stft of left and right channel
# because I think otherwise librosa will recompute the stft numerous times
class InputFeatures:
    """ 
    A class for objects, that extract and hold all needed features

    - rms: int --> overall energy contained in the input
    - rms_left: int --> energy in the left channel
    - rms_right: int --> energy in the right channel
    - dynamic_range: int --> difference between min and max amplitude
    - density: int --> measurement of how populated the input is
    - mid: int --> center channel signal
    - side: int --> side channel signal
    - spectral_centroid: int --> center of spectral gravity
    - spectral_flatness: int --> flat (noisy) vs non-flat (tonal) indication
    - spectral_rolloff: int --> threshold for majority of energy contained in lower subbands
    - spectral_spread: int --> instantaneous bandwidth as std derivation around spectral centroid
    """    

    def __init__(self, y: np.ndarray, sr: float):
        """
        Initializes the InputFeatures class.

        Parameters:
            y: The input audio signal, a 2D numpy array with shape (2, n_samples).
            sr: The sample rate of the audio signal.
        """
        # 1) start with some preprocessing
        if y.ndim != 2 or y is None:
            raise ValueError("Input array not applicable.")
        
        # Store the original audio signal for comparative analysis
        self.audio = y

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.instance_file_path = os.path.join(script_dir, '../ASP/instance.lp')
        
        # preprocess gammatone filterbank as they are audio independent and should be used over consecutive artifact analysis runs
        self.filterbank_low, self.filterbank_mid = ia.generate_gammatone_filterbank()
        
        # cache mel_spectrogram for comparative analysis and to avoid multiple STFT-computations
        # it will be used in the artifact analysis
        self.mel_left, self.mel_right = ia.compute_STFT(y=y, mode="mel")
        self.stft_left, self.stft_right = ia.compute_STFT(y=y, mode="regular")

        # 2) Extract instantaneous features 
        # compute rms related features
        rms, self.rms_mean, self.rms_channel_balance = ia.rms_features(y)
        self.dynamic_range = ia.compute_dynamic_rms(rms)
        self.density = (100 - self.dynamic_range) * self.rms_mean
        
        # compute spectral features
        self.spectral_centroid, centroid_l, centroid_r = ia.mean_spectral_centroid(S_l=self.stft_left, S_r=self.stft_right, sr=sr)
        self.spectral_flatness = ia.custom_flatness(S=np.abs(np.vstack([self.stft_left,self.stft_right])))
        self.spectral_spread = ia.spectral_spread(S_l=self.stft_left, S_r=self.stft_right, sr=sr, centroid_left=centroid_l, centroid_right=centroid_r)

        # compute mid-side values
        self.mid, self.side = ia.mid_side(y)

    def create_instance(self) -> Optional[str]:
        """Creation of an instance string describing our input for ASP guessing
        
        Returns:
            str: The instance string for ASP
        """

        instance = f"""
rms({int(self.rms_mean)}).
rms_channel_balance({int(self.rms_channel_balance)}).
dr({int(self.dynamic_range)}).
density_population({int(self.density)}).
mid({int(self.mid)}).
side({int(self.side)}).
spectral_centroid({int(self.spectral_centroid)}).
spectral_flatness({int(self.spectral_flatness)}).
spectral_spread({int(self.spectral_spread)})."""
        AspHandler.write_instance(instance, self.instance_file_path)

        return instance