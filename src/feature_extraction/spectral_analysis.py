import numpy as np
from scipy import signal

def estimate_t60(audio, sr):
    """
    Estimate T60 from a reverberant speech signal using energy decay analysis.
    
    Args:
        audio: np.array of audio samples
        sr: Sample rate in Hz
    Returns:
        Estimated T60 in seconds
    """
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Parameters
    window_size = int(0.05 * sr)  # 50ms windows
    hop_size = window_size // 2
    
    # Calculate energy in each window
    energy = []
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i + window_size]
        energy.append(np.sum(window ** 2))
    
    energy = np.array(energy)
    
    # Convert to dB
    energy_db = 10 * np.log10(energy + 1e-10)
    
    # Find peaks to detect decay curves
    peaks = signal.find_peaks(energy_db, distance=10)[0]
    
    # Analyze decay rates after peaks
    decay_rates = []
    for peak in peaks:
        if peak + 20 < len(energy_db):
            decay_curve = energy_db[peak:peak + 20]
            # Fit line to decay curve
            x = np.arange(len(decay_curve))
            A = np.vstack([x, np.ones(len(x))]).T
            slope, _ = np.linalg.lstsq(A, decay_curve, rcond=None)[0]
            decay_rates.append(-slope)
    
    if len(decay_rates) == 0:
        return None
        
    # Calculate T60 from median decay rate
    median_decay = np.median(decay_rates)
    t60 = 60 / (median_decay * (sr / hop_size))
    
    return t60

def estimate_drr(audio, sr):
    """
    Estimate DRR from a reverberant speech signal.
    
    Args:
        audio: np.array of audio samples
        sr: Sample rate in Hz
    Returns:
        Estimated DRR in dB
    """
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        
    # Parameters
    early_time = 0.05  # 50ms for early reflections
    early_samples = int(early_time * sr)
    
    # Calculate energy in early and late parts
    early_energy = np.sum(audio[:early_samples] ** 2)
    late_energy = np.sum(audio[early_samples:] ** 2)
    
    # Calculate DRR
    drr = 10 * np.log10(early_energy / (late_energy + 1e-10))
    
    return drr