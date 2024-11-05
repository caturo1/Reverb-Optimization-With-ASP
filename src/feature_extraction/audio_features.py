from typing import Any


class AudioFeatures:
    def __init___(self):
        #self.mean_spectral_centroid: float
        #self.mean_spectral_flatness: float
        #self.spectral_rolloff: float
        #self.median_spectral_contrast: float
        #self.spectral_flatness: float
        self.dynamic_range: int
        self.mean_rms: int
        self.density: int

def to_clingo(self) -> str:
    return f"{self.dynamic_range},{self.mean_rms})."


# define __getattr__ and __setattr__ to allow for dynamic attribute access