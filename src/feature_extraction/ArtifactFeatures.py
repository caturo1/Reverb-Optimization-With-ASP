from . import input_analysis as ia
from . import artifact_analysis as aa
import numpy as np

class ArtifactFeatures:
    """"
    Class that represents the result of an artifact analysis
    of a corresponding reverbrated audio.
    """

    def __init__(self, 
                y_org: np.ndarray, 
                y_proc: np.ndarray,
                mel_l_org: np.ndarray,
                mel_r_org: np.ndarray,
                filter_bank_num_low: list,
                filter_bank_num_mid: list): 
        
        # to save computation, store spectrogram
        self.mel_left,self.mel_right = ia.compute_STFT(y_proc, mode="mel")

        self.clipping_l = aa.clipping_analyzer(y_proc[0])
        self.clipping_r = aa.clipping_analyzer(y_proc[1])

        # for differntial analysis and both cahnnels
        self.b2mR = aa.muddiness_analyzer(y_org=y_org, y_proc=y_proc, filter_bank_num_low=filter_bank_num_low, filter_bank_num_mid=filter_bank_num_mid)
        self.cc = aa.cross_correlation(y=y_proc)

        # differential analysis for both channels
        """
        self.den_stability_differential_l, self.den_diff_differential_l = aa.spectral_density(self.mel_left, mel_l_org)
        self.den_stability_differential_r, self.den_diff_differential_r = aa.spectral_density(self.mel_right, mel_r_org)

        # differential analysis for both channels
        self.clustering_differential_l = aa.spectral_clustering(self.mel_left, mel_l_org)
        self.clustering_differential_r = aa.spectral_clustering(self.mel_right, mel_r_org)
        """

        # analysis for both channels
        # we can roughly estimate that the ringing will be in [0,528]
        # where 528 will basically never happen. Thresholds will have to be defined later on
        self.ringing_l = aa.ringing(self.mel_left, mel_l_org)
        self.ringing_r = aa.ringing(self.mel_right, mel_r_org)

    def to_string(self):
        """
        Just print the object toString
        """
        print(f"Bass to mid: {self.b2mR}\n"
              f"clipping: {self.clipping_l, self.clipping_r}\n"
              f"Cross-correlation: {self.cc}\n"
              f"ringing: {self.ringing_l, self.ringing_r}")