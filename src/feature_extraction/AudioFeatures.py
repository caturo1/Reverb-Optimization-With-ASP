import input_analysis

class AudioFeatures:
    """ rms_left: int
        rms_right: int
        dynamic_Range: int
        spectral_centroid: int
        spectral_flatness: int
        spectral_rolloff: int
        mid: int
        side: int
        density: int
        is_mono: int
        flatness: int
        rolloff: int
    """
    
    def __init__(self, y: np.ndarray, sr: int, S):
        #initialize all the features based on y,sr,STFT
    



