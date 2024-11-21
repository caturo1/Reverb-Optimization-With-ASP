import numpy as np
import input_analysis as ia

class AudioFeatures:
    """ 
    A class for objects, that extract and hold all needed features

    rms: int --> overall energy contained in the input
    rms_left: int --> energy in the left channel
    rms_right: int --> energy in the right channel
    dynamic_range: int --> difference between min and max amplitude
    density: int --> measurement of how populated the input is
    mid: int
    side: int
    spectral_centroid: int
    spectral_flatness: int
    spectral_rolloff: int
    """
    
    def __init__(self, y: np.ndarray, sr: int, S):
        self.rms, rms_mean, self.rms_l, self.rms_r = ia.rms_features(y)
        self.dynamic_range = ia.compute_dynamic_rms(self.rms)
        self.density = (100 - self.dyn_rms) * rms_mean
        self.mid, self.side = ia.mid_side(y)
        self.spectral_centroid = ia.mean_spectral_centroid(y)
        self.spectral_flatness = ia.mean_spectral_flatness(y)

