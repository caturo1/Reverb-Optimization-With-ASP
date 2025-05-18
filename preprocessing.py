import argparse
import numpy as np
import librosa
from typing import Tuple, Optional
import soundfile
import os
from pathlib import Path

RATE = 44_100

def load_audio(file: str) -> Tuple[np.ndarray, int]:
    """Load audio input as a stereo file"""
    y, sr = librosa.load(path=file, sr=RATE, mono=False, res_type="kaiser_best")
    y = to_stereo(y)
    return y, sr

def to_stereo(y: Optional[np.ndarray]) -> np.ndarray:
    """Create stereo signal by duplicating the mono signal if necessary"""
    if y is None:
        raise ValueError("Input signal cannot be None")
    
    # Handle different input shapes
    if y.ndim == 1:
        # Mono signal - duplicate to stereo
        return np.vstack((y, y))
    elif y.ndim == 2:
        if y.shape[0] == 1:
            # Single channel in first dimension
            return np.vstack((y[0], y[0]))
        elif y.shape[1] == 1:
            # Single channel in second dimension
            return np.vstack((y[:, 0], y[:, 0]))
        elif y.shape[0] == 2:
            # Already stereo, correct orientation
            return y
        elif y.shape[1] == 2:
            # Stereo but wrong orientation
            return y.T
        else:
            # Multi-channel audio, take first two channels
            if y.shape[0] < y.shape[1]:
                # Channels are in first dimension
                return y[:2, :]
            else:
                # Channels are in second dimension
                return y[:, :2].T
    else:
        raise ValueError(f"Unexpected audio shape: {y.shape}")

def amplitude_norm(y: np.ndarray) -> np.ndarray:
    """
    Amplitude normalization to ~ -0.9 dBFS, ensuring headroom

    Parameters:
        y: Input signal, stereo with shape (2, samples)
    
    Returns:
        y_norm: Normalised audio signal
    """
    peak_amplitude = np.max(np.abs(y))
    if peak_amplitude > 0:  # Avoid division by zero
        y_norm = y * 0.9 / peak_amplitude
    else:
        y_norm = y
    return y_norm

def length_norm(y: np.ndarray, sr: int, target_duration: float = 7.0) -> np.ndarray:
    """Normalize audio length to target duration"""
    target_samples = int(target_duration * sr)
    
    # Ensure correct shape (2, samples) for stereo
    if y.shape[0] != 2:
        raise ValueError(f"Expected stereo signal with shape (2, samples), got {y.shape}")
    
    current_samples = y.shape[1]
    
    if current_samples > target_samples:
        # Use middle section to avoid silence at start/end
        start = (current_samples - target_samples) // 2
        return y[:, start:start + target_samples]
    else:
        # Pad if too short
        pad_width = [(0, 0), (0, target_samples - current_samples)]
        return np.pad(y, pad_width, mode='constant')

def process_audio_files(input_dir: str, output_dir: str):
    """Process all audio files in input directory and save to output directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define supported audio extensions
    audio_extensions = {'.wav', '.flac', '.mp3'}
    
    processed_count = 0
    error_count = 0
    
    for file_path in input_path.rglob('*'):
        if file_path.suffix.lower() in audio_extensions:
            relative_path = file_path.relative_to(input_path)
            output_file_path = output_path / relative_path.with_suffix('.wav')
            
            # Create subdirectories if necessary
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                print(f"Processing: {relative_path}")
                
                # Load and process audio
                y, sr = load_audio(str(file_path))
                y_norm = amplitude_norm(y)
                y_fin = length_norm(y_norm, sr)
                
                # Write normalized audio - transpose back for soundfile
                soundfile.write(file=str(output_file_path), data=y_fin.T, samplerate=RATE)
                processed_count += 1
                print(f"  Success: {output_file_path}")
                
            except Exception as e:
                print(f"Error processing {relative_path}: {str(e)}")
                error_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Errors encountered: {error_count} files")

if __name__ == "__main__":
    # parse CLI arguments
    parser = argparse.ArgumentParser(description="Batch normalize audio files for benchmarking")
    parser.add_argument("--input", default="bench_data", help="Input directory with audio files (default: bench_data)")
    parser.add_argument("--output", default="bench_res", help="Output directory for normalized audio (default: bench_res)")
    
    args = parser.parse_args()
    
    # Process all audio files in the input directory
    process_audio_files(args.input, args.output)