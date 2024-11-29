import os
import numpy as np
import sys
from AspHandler import *
import input_analysis as ia
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
        # store path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.instance_file_path = os.path.join(script_dir, '../ASP/instance.lp')
        
        # store mel_spectrogram for comparative analysis
        self.mel_left = ia.compute_STFT(y[0], mode="mel")
        self.mel_right = ia.compute_STFT(y[1], mode="mel")

        # store feats
        rms, self.rms_mean, self.rms_channel_balance = ia.rms_features(y)
        self.dynamic_range = ia.compute_dynamic_rms(rms)
        self.density = (100 - self.dynamic_range) * self.rms_mean
        self.mid, self.side = ia.mid_side(y)
        self.spectral_centroid, centroid_l, centroid_r = ia.mean_spectral_centroid(y=y, sr=sr)
        self.spectral_flatness = ia.mean_spectral_flatness(y=y)
        self.spectral_spread = ia.spectral_spread(S_l=self.stft_left, S_r=self.stft_right, sr=sr, centroid_left=centroid_l, centroid_right=centroid_r)


    def create_instance(curr) -> Optional[str]:
        """Creation of an instance string describing our input for ASP guessing"""
        
        instance = f"""
rms({int(curr.rms_mean)}).
rms_channel_balance({int(curr.rms_channel_balance)}).
dr({int(curr.dynamic_range)}).
density_population({int(curr.density)}).
mid({int(curr.mid)}).
side({int(curr.side)}).
spectral_centroid({int(curr.spectral_centroid)}).
spectral_flatness({int(curr.spectral_flatness)}).
spectral_spread({int(curr.spectral_spread)})."""
        
        AspHandler.write_instance(instance, curr.instance_file_path)

    def read_input(self, sample: str):
        """Read input and analyze features"""
        try: 
            y, sr = ia.load_audio(sample)
            
        except Exception as e:
            print(f"Error {e} processing input audio")
            sys.exit(1)

        current = InputFeatures(y=y, sr=sr)
        self.create_instance(current)

        return current