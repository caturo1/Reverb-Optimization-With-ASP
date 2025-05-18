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

        # clipping analysis of both channels
        self.clipping_l = aa.clipping_analyzer(y_proc[0])
        self.clipping_r = aa.clipping_analyzer(y_proc[1])

        # for differntial analysis and both cahnnels
        # we can use different methods
        # 1) we can use the gammatone filterbank -> precise but slow
        # 2) we can use the mel spectrum -> fast but not precise
        self.b2mR = aa.muddiness_analyzer_gammatone(y_org=y_org, y_proc=y_proc, filter_bank_num_low=filter_bank_num_low, filter_bank_num_mid=filter_bank_num_mid)
        #self.b2mR = aa.muddiness_analyzation(mel_S=self.mel_left, mel_org=mel_l_org)
        self.cc = aa.cross_correlation(y=y_proc)

        # resonance analysis for both channels
        # we can roughly estimate that the ringing will be in [0,528]
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