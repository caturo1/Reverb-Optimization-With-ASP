"""
Signal Generation Benchmark Script

Generates variable test signals and runs an Optimizer, measuring solve time for each signal.
Run from root with: 'python -m src.test'
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import soundfile as sf
from scipy import signal
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from typing import Union, Tuple, Optional

class SignalGenerator:
    """
    Generates test signals for benchmarking purposes.
    
    * Writes signal into benchmark directory
    * Has different control parameters to explore edge cases
        - duration: Determines the duration of the signal
        - frequency: Determines the frequency of the generated signal
        - complex: If set, generates a signal with varying amplitude and 
        - dyn_range_control: if set, limited dynamic range and
        - composite: if set, a harmonically complex, convoluted signal and
        - phase_offset: a stereo signal with a maximum offset of 90° and
        - modulated: FM modulation to simulate frequency variations (measured by spectral spread and flatness)
    * Otherwise a simple sine wave will be generated, if that is what we want to examine

    These configurations tap into different areas of the search space of the RevOp, so we can evaluate how
    any of these might influence the search space.
    
    The parenting benchmark script will store those parameters for every run to map them to clingo and time stats
    """
    
    # Constants
    SAMPLE_RATE = 44_100
    BIT_DEPTH = 16
    PERIOD = 1.0 / SAMPLE_RATE
    #LOWER_BOUND = -(2**BIT_DEPTH) / 2
    #UPPER_BOUND = ((2**BIT_DEPTH) / 2) - 1
    MAX_PHASE_OFFSET = np.pi / 2
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize SignalGenerator with output directory.
        
        Args:
            output_dir: Directory where generated signals will be saved
        """
        self.output_dir = Path(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        #random.seed(132413)  # For reproducibility

    
    def generate_signal(
        self,
        name: str,
        amplitude: Optional[int],
        frequency: float,
        duration: float = 1.0,
        complex: bool = True,
        phase_offset: float = 0.0,
        modulated: bool = True,
        mod_frequency: float = 0.12,
        mod_depth: float = 100.0
    ) -> Tuple[np.ndarray, int]:
        """
        Generate a test signal with specified parameters.
        
        Args:
            name: Output filename
            frequency: Signal frequency in Hz
            duration: Signal duration in seconds
            complex: If True, generates complex signal with varying amplitudes
            phase_offset: Phase offset in radians
            
        Returns:
            Tuple of (generated signal array, sample rate)
        """
        n_samples = int(duration * self.SAMPLE_RATE)
        time = np.arange(n_samples) * self.PERIOD

        if complex:
            signal = self._generate_complex_signal(
                amp=amplitude,
                n_samples=n_samples, 
                frequency=frequency, 
                time=time, 
                phase_offset=phase_offset,
                modulated=modulated,
                mod_frequency=mod_frequency,
                mod_depth=mod_depth)
        else:
            signal = self._generate_simple_signal(duration, frequency, phase_offset)
            
        output_path = self.output_dir / f"{name}.wav"
        # print(output_path)
        sf.write(
            file=output_path,
            data=signal,
            samplerate=self.SAMPLE_RATE,
            format='WAV',
            subtype="FLOAT"
        )
        
        return signal, self.SAMPLE_RATE

    def _frequency_modulation(
        self,
        time: np.ndarray,
        carrier_freq : np.ndarray,
        mod_frequency: float ,
        beta: float):
        """
        Apply frequency modulation to a carrier frequency.
        Formula: f(t) = A*sin(w_c * t + beta * sin(w_m * t))
                w_c = 2 * pi * f_c
                w_m = 2 * pi * f_m

        Args:
            time: Time array
            carrier_freq: The carrier frequency to modulate
            mod_frequency: Frequency of the modulating signal
            beta: Modulation index (depth); controls amp of modulating osc, thus swing of carrier freq.
            
        Returns:
            res: Frequency modulated signal

        Note: 
            If beta != 0,a multitude of sidebands appear at multiples of +-f_m, 
            with often the energy of the carrierer being dispersed into the sidebands.
            Otherwise, res is a pure sine.
            In FM modulation, the phase velocity advances and retreats, alternatively
            increasing and decreasing its frequency slightly.
        """
        
        mod_sig = np.sin(2 * np.pi * mod_frequency * time)
        phi = 2 * np.pi * carrier_freq * time + beta * carrier_freq * np.cumsum(mod_sig) * self.PERIOD

        res = np.sin(phi)
        return res
        

    def _generate_complex_signal(
        self,
        n_samples: int,
        frequency: float,
        time: np.ndarray,
        phase_offset: float,
        amp: Optional[int],
        stereo : bool = True,
        dyn_range_control : bool = False,
        composite : bool = True,
        modulated: bool = True,
        mod_frequency: float = 5.0,
        mod_depth: float = 1.0
    ) -> np.ndarray:
        """Generate a complex signal with varying amplitudes."""
        # Limit number of amplitude transitions
        n_anchors = random.randrange(0, min(self.SAMPLE_RATE // 100, 20))
        if not (-self.MAX_PHASE_OFFSET < phase_offset < self.MAX_PHASE_OFFSET):
            phase_offset = random.uniform(self.MAX_PHASE_OFFSET, self.MAX_PHASE_OFFSET)
        
        def generate_amp_envelope():
            # Generate random amplitude anchor points
            n_points = max(2, n_anchors)

            # if we want a small dynamic range, we will just generate 
            # amp values in a fraction of the possible range
            if dyn_range_control:
                anchor_values = [random.uniform(-1.0 / 4, 1.0 / 4) for _ in range(n_points)]

            else:
                anchor_values = [random.uniform(-0.85, 0.85) for _ in range(n_points)]



            # for every amp value, assign a random time value
            # but adhere to a minimal spacing to not create artifacts
            # we can use this to consciously create transients as well
            anchor_positions = np.linspace(0, n_samples-1, n_points, dtype=int)

            # Using cobic splines, interpolate between the points   
            # cs = CubicSpline(anchor_positions, anchor_values)
            envelope = np.interp(np.arange(len(time)), anchor_positions, anchor_values)

            """
            plt.figure()
            plt.plot(envelope)
            plt.title("Envelope Shape")
            plt.show()

            print("Anchor values:", anchor_values)
            print("Time points:", anchor_positions)
            print("Shape min/max:", np.min(cs(time)), np.max(cs(time)))
            
            return cs(time)
            """

            return envelope

        if not stereo:
            # in case of mono signal use the same signal for both channels
            shape = generate_amp_envelope()

            y = shape * np.sin(2 * np.pi * frequency * time)
            return np.vstack((y, y)).T
        
        # otherwise generate 2 rand amplitude envelopes for both channels
        if not isinstance(amp, int):
            print("Generating complex amplitude envelope")
            shape_l = generate_amp_envelope()
            shape_r = generate_amp_envelope()
        else:
            print("Using simple amplitude scalars")
            shape_l = amp
            shape_r = amp
            

        if composite:
            # convolute sin, saw and square waves to create a complex signal
            # with scaled shape
            def compose_signal(shape):

                if modulated:
                    base_sig = self._frequency_modulation(
                        mod_frequency=mod_frequency, 
                        beta=mod_depth, 
                        time=time, 
                        carrier_freq=frequency)
                    base_sig_norm = base_sig / (np.max(np.abs(base_sig)) + 1e-10)
                
                else: 
                    base_sig = np.sin(2 * np.pi * frequency * time)
                
                saw = 0.5 * signal.sawtooth(2 * np.pi * frequency * time)
                triangle = 0.25 * signal.sawtooth(2 * np.pi * frequency * time, width=0.5)
                independent_square = np.sign(np.sin(2 * np.pi * frequency * time))
                square = 0.25 * np.sign(base_sig_norm)
                sq1 = 0.05 * signal.waveforms._waveforms.square(time)

                #y = base_sig * saw * square
                # convolution of signals
                y = signal.fftconvolve(in1 = base_sig_norm, in2 = saw * square, mode='same')
                y = y / (np.max(np.abs(y)) + 1e-10)
                #y = np.convolve(a = base_sig, v = saw * square)
                
                """
                print("Base signal min/max:", np.min(base_sig_norm), np.max(base_sig_norm))
                print("Saw min/max:", np.min(saw), np.max(saw))
                print("Square min/max:", np.min(square), np.max(sq1))
                print("Before envelope min/max:", np.min(y), np.max(y))
                
                plt.figure()
                plt.plot(y)
                plt.title("Final Signal")
                plt.show()
                """


                # apply amplitude envelope
                # print(shape)
                # maybe clip y to its original shape to multiply properly with shape.
                # but the convolution should keep the dimensions of in1
                y = y * shape
                #print("After envelope min/max:", np.min(y), np.max(y))


                # make sure the convolution doesn't lead to unintentional clippnig
                if dyn_range_control:
                    return np.clip(a=y, a_min= -1.0 / 8, a_max= 1.0 / 8)
                
                return np.clip(a=y, a_min= -1.0, a_max= 0.9)

            y_left = compose_signal(shape=shape_l)
            y_right = compose_signal(shape=shape_r)

            return np.vstack((y_left, y_right)).T

        # generate stereo signal using those two amplitude variations
        y_left = shape_l * np.sin(2 * np.pi * frequency * time)
        y_right = shape_r * np.sin(2 * np.pi * frequency * time + phase_offset)
        
        return np.vstack((y_left, y_right)).T


    def _generate_simple_signal(
        self,
        duration: float,
        frequency: float,
        phase_offset: float
    ) -> np.ndarray:
        """Generate a simple signal with constant amplitude per second."""
        time = np.arange(0, 1, self.PERIOD, dtype=np.float32)
        signal = np.zeros_like(time)

        if not (-self.MAX_PHASE_OFFSET < phase_offset < self.MAX_PHASE_OFFSET):
            phase_offset = random.randrange(-self.MAX_PHASE_OFFSET, self.MAX_PHASE_OFFSET)
        
        for _ in range(int(duration)):
            amplitude = random.randrange(self.LOWER_BOUND, self.UPPER_BOUND)
            signal += amplitude * np.sin(2 * np.pi * frequency * time + phase_offset)
        
        return signal

def main(argv): 
    """Main entry point for signal generation."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-f", "--frequency",
        help="Specify signal frequency as integer",
        type=int
    )

    ap.add_argument(
        "-d", "--duration",
        help="Maximum signal length in seconds",
        type=int
    )

    ap.add_argument(
        "-p", "--phase",
        help="Phase offset for stereo effect as float",
        type=float
    )

    ap.add_argument(
        "-mf", "--modFreq",
        help="Modulating frequency",
        type=int
    )

    ap.add_argument(
        "-md", "--modDepth",
        help="Modulation depth for FM Synthesis",
        type=float
    )

    ap.add_argument(
        "-n", "--name",
        help="name of the generated signal",
        type=str
    )

    ap.add_argument(
        "-r", "--run",
        help="number of the current run",
        type=int
    )

    ap.add_argument(
        "-a", "--amplitude",
        help="Amplitude value",
        type=int
    )
    
    ap.add_argument(
        "-m", "--mode",
        help="Specify, if the signal should be complex or not",
        type=bool,
    )
    
    args = ap.parse_args()

    generator = SignalGenerator(output_dir="./data/testing")
    
    frequency = args.frequency
    duration = args.duration
    n_run = args.run
    amp = args.amplitude
    phase_offset = np.pi / float(n_run)
    #convert radians to degree for clarity in stats
    deg = phase_offset * (180 / np.pi)
    #modulating freq: preferably low (<0.2) values for refined frequency variations
    mf = args.modFreq / 100
    #modulation depth: higher values for stronger carrier modulation
    md = args.modDepth / 100
    name = args.name
    #name = f"test_{frequency}Hz_{duration}s_{phase_offset}offset_{n_run}_run"
    
    stats = {
        'frequency' : frequency,
        'duration' : duration,
        'phase_offset' : float(deg), # convert, because numpy float gave me some problems
        'modulating_frequency' : mf,
        'modulation_index' : md,
        'amplitude' : amp
    }

    result_dir = Path("./benchmark/results")
    # also taken care of in the batch script
    result_dir.mkdir(parents=True, exist_ok=True)

    generator.generate_signal(
        name=name,
        amplitude=amp,
        frequency=frequency,
        duration=duration,
        phase_offset=phase_offset,
        mod_frequency=mf,
        mod_depth=md
    )

    json_path = result_dir / f"audio_test_{n_run}.json"
    with open(json_path, 'w') as f:  # Note the 'w' mode for writing
        json.dump(stats, f, indent=4)
            
if __name__ == "__main__":
    main(sys.argv[1:])