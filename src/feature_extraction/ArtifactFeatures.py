import sys
import input_analysis as ia
import librosa
from typing import Optional
import artifact_analysis as aa
import numpy as np

class ArtifactFeatures:
    """"
    Class that represents the result of an artifact analysis
    of a corresponding reverbrated audio.
    """

    def __init__(self, 
                 y: np.ndarray, 
                 mel_l_org: np.ndarray,
                 mel_r_org: np.ndarray): 
        
        # to save computation, store spectrogram

        self.mel_left,self.mel_right = ia.compute_STFT(y, mode="mel")

        self.clipping_l = aa.clipping_analyzer(y[0])
        self.clipping_r = aa.clipping_analyzer(y[1])

        # for differntial analysis and both cahnnels
        bass_to_mid_ratio_l = aa.muddiness_analyzation(mel_S=self.mel_left)
        bass_to_mid_ratio_r = aa.muddiness_analyzation(mel_S=self.mel_right)
        bass_to_mid_ratio_l_org = aa.muddiness_analyzation(mel_S=mel_l_org)
        bass_to_mid_ratio_r_org = aa.muddiness_analyzation(mel_S=mel_r_org)

        self.b2mR_L = bass_to_mid_ratio_l - bass_to_mid_ratio_l_org
        self.b2mR_R = bass_to_mid_ratio_r - bass_to_mid_ratio_r_org
        
        self.cc = aa.cross_correlation(y)

        # differential analysis for both channels
        self.den_stability_differential_l, self.den_diff_differential_l = aa.spectral_density(self.mel_left, mel_l_org)
        self.den_stability_differential_r, self.den_diff_differential_r = aa.spectral_density(self.mel_right, mel_r_org)

        # differential analysis for both channels
        self.clustering_differential_l = aa.spectral_clustering(self.mel_left, mel_l_org)
        self.clustering_differential_r = aa.spectral_clustering(self.mel_right, mel_r_org)

        # analysis for both channels
        # we can roughly estimate that the ringing will be in [0,528]
        # where 528 will basically never happen. Thresholds will have to be defined later on
        self.ringing_l = aa.ringing(self.mel_left, mel_l_org)
        self.ringing_r = aa.ringing(self.mel_right, mel_r_org)
