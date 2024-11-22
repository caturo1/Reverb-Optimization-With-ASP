import numpy as np
import input_analysis as ia
from typing import Optional
class AudioFeatures:
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
    
    def __init__(self, y: np.ndarray, sr: float, S: Optional[np.ndarray]):
        rms, self.rms_mean, self.rms_channel_balance = ia.rms_features(y)
        self.dynamic_range = ia.compute_dynamic_rms(rms)
        self.density = (100 - self.dynamic_range) * self.rms_mean
        self.mid, self.side = ia.mid_side(y)
        self.cross_corr = ia.cross_correlation(y)
        self.spectral_centroid, centroid_l, centroid_r = ia.mean_spectral_centroid(y=y, sr=sr)
        self.spectral_flatness = ia.mean_spectral_flatness(y=y)
        self.spectral_spread = ia.spectral_spread(S=S, sr=sr, centroid_left=centroid_l, centroid_right=centroid_r)

