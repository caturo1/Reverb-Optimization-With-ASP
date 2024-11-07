# nice but can use dict and parse this directly into Jason
class AudioFeatures:

    def __init___(
            self, dynamic_range, 
            mean_rms, density, 
            spec_cent):
        #self.mean_spectral_centroid: int
        #self.mean_spectral_flatness: int
        #self.spectral_rolloff: int
        #self.median_spectral_contrast: int
        #self.spectral_flatness: int
        self.mean_spectral_centroid = dynamic_range
        self.mean_rms = mean_rms
        self.density = density
        self.mean_spectral_centroid = spec_cent