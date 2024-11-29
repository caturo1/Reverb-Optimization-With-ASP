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
                 sr: float, 
                 mel_l_org: Optional[np.ndarray],
                 mel_r_org: Optional[np.ndarray]): 
        
        # to save computation, store spectrogram
        self.mel_left = ia.compute_STFT(y[0], mode="mel")
        self.mel_right = ia.compute_STFT(y[1], mode="mel")

        self.clipping = aa.clipping_analyzer(y)

        # for differntial analysis and both cahnnels
        self.bass_ratio_l, self.mid_ratio_l, self.bass_to_mid_ratio_l = aa.muddiness_analyzation(mel_S=self.mel_l)
        self.bass_ratio_r, self.mid_ratio_r, self.bass_to_mid_ratio_r = aa.muddiness_analyzation(mel_S=self.mel_r)
        self.bass_ratio_l_org, self.mid_ratio_l_org, self.bass_to_mid_ratio_l_org = aa.muddiness_analyzation(mel_S=mel_l_org)
        self.bass_ratio_r_org, self.mid_ratio_r_org, self.bass_to_mid_ratio_r_org = aa.muddiness_analyzation(mel_S=mel_r_org)
        
        self.cc = aa.cross_correlation(y)
        
        # differential analysis for both channels
        self.den_ratio_differential_l, self.den_diff_differential_l = aa.spectral_density(self.mel_l, mel_l_org)
        self.den_ratio_differential_r, self.den_diff_differential_r = aa.spectral_density(self.mel_r, mel_r_org)

        # differential analysis for both channels
        self.regularity_differential_l, self.clustering_differential_l, self.resonance_differential_l, self.resonance_score_l, self.clustering_score_l = aa.spectral_clustering(self.mel_l, mel_l_org)
        self.regularity_differential_r, self.clustering_differential_r, self.resonance_differential_r, self.resonance_score_r, self.clustering_score_r = aa.spectral_clustering(self.mel_l, mel_l_org)

        # analysis for both channels
        self.lingering_l = aa.ringing(self.mel_l, sr)
        self.lingering_r = aa.ringing(self.mel_r, sr)

    def create_instance(self) -> str:
        """Creates ASP instance string from artifact features."""
        return f"""
clipping({int(self.clipping)}).

bass_ratio_left({int(self.bass_ratio_l)}).
mid_ratio_left({int(self.mid_ratio_l)}).
bass_to_mid_ratio_left({int(self.bass_to_mid_ratio_l)}).

bass_ratio_right({int(self.bass_ratio_r)}).
mid_ratio_right({int(self.mid_ratio_r)}).
bass_to_mid_ratio_right({int(self.bass_to_mid_ratio_r)}).

bass_ratio_left_org({int(self.bass_ratio_l_org)}).
mid_ratio_left_org({int(self.mid_ratio_l_org)}).
bass_to_mid_ratio_left_org({int(self.bass_to_mid_ratio_l_org)}).

bass_ratio_right_org({int(self.bass_ratio_r_org)}).
mid_ratio_right_org({int(self.mid_ratio_r_org)}).
bass_to_mid_ratio_right_org({int(self.bass_to_mid_ratio_r_org)}).

cross_correlation({int(self.cc)}).

density_ratio_differential_left({int(self.den_ratio_differential_l)}).
density_difference_differential_left({int(self.den_diff_differential_l)}).
density_ratio_differential_right({int(self.den_ratio_differential_r)}).
density_difference_differential_right({int(self.den_diff_differential_r)})."""
    
    
    def read_output(self, sample: str):
        """Read processed audio and analyze features"""
        try: 
            y, sr = ia.load_audio(sample)
            
        except Exception as e:
            print(f"Error {e} processing input audio")
            sys.exit(1)

        return ArtifactFeatures(
            y=y, sr=sr, 
            mel_l_org=self.input_features.mel_left, mel_r_org=self.input_features.mel_right
            )
