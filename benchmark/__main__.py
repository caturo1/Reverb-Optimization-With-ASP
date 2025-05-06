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
    MAX_PHASE_OFFSET = (np.pi / 2)
    
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
        amplitude: Optional[Tuple[int]],
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

        phase_offset = np.clip(phase_offset, 0, self.MAX_PHASE_OFFSET)

        if complex:
            signal = self._generate_complex_signal(
                amp_dr=amplitude,
                n_samples=n_samples, 
                frequency=frequency, 
                time=time, 
                phase_offset=phase_offset,
                modulated=modulated,
                mod_frequency=mod_frequency,
                mod_depth=mod_depth)
        else:
            signal = self._generate_simple_signal(
                duration=duration,
                frequency=frequency,
                phase_offset=phase_offset,
                dyn_range=amplitude)
            
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

    def compose_signal(
        self, 
        shape, 
        modulated, 
        mod_frequency, 
        mod_depth,
        time,
        frequency):

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
        square = 0.25 * np.sign(base_sig_norm) if modulated else 0.25 * np.sign(base_sig)
        sq1 = 0.05 * signal.waveforms._waveforms.square(time)

        #y = base_sig * saw * square
        # convolution of signals
        y = signal.fftconvolve(in1 = base_sig_norm if modulated else base_sig, in2 = saw * square, mode='same')
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
        return np.clip(a=y, a_min= -1.0, a_max= 0.9)

    def generate_amp_envelope(
        self,
        n_anchors : int,
        n_samples : int,
        time : np.ndarray,
        amp_dr : Tuple[float],
        dyn_range_control : bool = False,
    ):
        """
        Generate amplitude envelope.
        We can control the dynamic range by assigning
        a range of values, where we will randomly pick amplitude values from.
        If we want static, simple signals, upper and lower bound can just be the same.
        """
        # Generate random amplitude anchor points
        n_points = max(2, n_anchors)

        # if we want a small dynamic range, we will just generate 
        # amp values in a fraction of the possible range
        if dyn_range_control:
            anchor_values = [random.uniform(amp_dr[0], amp_dr[1]) for _ in range(n_points)]

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
        amp_dr: Optional[Tuple[float]],
        stereo : bool = True,
        dyn_range_control : bool = False,
        modulated: bool = True,
        mod_frequency: float = 5.0,
        mod_depth: float = 1.0
    ) -> np.ndarray:
        """Generate a complex signal with options:
        - randomized amplitude envelope
        - control over randomization with specified dynamic range
        - composite signal waves (saw, square, sin, triangle)
        - option between stereo/mono
        - varying degree of phase offset, thus stereo separation between channels
        - frequency modulation using different carrier frequencies and modulation depth

        These parameters explore most of the features relevant to clingo.
        """
        # Limit number of amplitude transitions
        n_anchors = random.randrange(0, min(self.SAMPLE_RATE // 100, 20))
        if not (-self.MAX_PHASE_OFFSET < phase_offset < self.MAX_PHASE_OFFSET):
            phase_offset = random.uniform(self.MAX_PHASE_OFFSET, self.MAX_PHASE_OFFSET)

        if not stereo:
            # in case of mono signal use the same signal for both channels
            print("Generating complex mono signal with dynamic range")
            shape = self.generate_amp_envelope(
                n_anchors=n_anchors,
                dyn_range_control=dyn_range_control,
                n_samples=n_samples,
                time=time,
                amp_dr=amp_dr)

            y = self.compose_signal(
                shape=shape,
                modulated=modulated, 
                mod_frequency=mod_frequency, 
                mod_depth=mod_depth,
                time=time,
                frequency=frequency)
            
            return np.vstack((y, y)).T
        
        # otherwise generate 2 rand amplitude envelopes for both channels
        print("Generating complex stereo signal with dynamic range")
        shape_l = self.generate_amp_envelope(
            n_anchors=n_anchors,
            dyn_range_control=dyn_range_control,
            n_samples=n_samples,
            time=time,
            amp_dr=amp_dr)
        shape_r = self.generate_amp_envelope(
            n_anchors=n_anchors,
            dyn_range_control=dyn_range_control,
            n_samples=n_samples,
            time=time,
            amp_dr=amp_dr)


        y_left = self.compose_signal(
            shape=shape_l, 
            modulated=modulated, 
            mod_frequency=mod_frequency, 
            mod_depth=mod_depth,
            time=time,
            frequency=frequency)
        y_right = self.compose_signal(
            shape=shape_r,
            modulated=modulated, 
            mod_frequency=mod_frequency, 
            mod_depth=mod_depth,
            time=time,
            frequency=frequency)
        
        return np.vstack((y_left, y_right)).T


    def _generate_simple_signal(
        self,
        duration: float,
        frequency: float,
        amplitude_range: Tuple[float, float],
        phase_offset: float = 0,
        fade_edges: bool = True,
        signal_type: str = "sine"
    ) -> np.ndarray:
        """
        Generate a simple test signal with controlled parameters.
        
        Parameters:
        -----------
        duration : float
            Duration of the signal in seconds
        frequency : float
            Base frequency of the signal in Hz
        amplitude_range : Tuple[float, float]
            Range of amplitude values (min, max) for dynamic amplitude changes
            Use (x, x) for static amplitude of value x
        phase_offset : float
            Phase offset between left and right channels in radians
        fade_edges : bool
            Whether to apply fade in/out to avoid clicks at signal edges
        signal_type : str
            Type of signal to generate: "sine", "square", "sawtooth", "noise", "sweep"
            
        Returns:
        --------
        np.ndarray
            Stereo signal array of shape (samples, 2)
        """
        # Ensure phase offset is within bounds
        if not (-self.MAX_PHASE_OFFSET < phase_offset < self.MAX_PHASE_OFFSET):
            phase_offset = random.uniform(-self.MAX_PHASE_OFFSET, self.MAX_PHASE_OFFSET)
        
        # Generate time array for the entire signal duration
        n_samples = int(duration * self.SAMPLE_RATE)
        time = np.arange(n_samples) * self.PERIOD
        
        # For dynamic amplitude changes, create a smooth amplitude envelope
        if amplitude_range[0] != amplitude_range[1]:
            # Generate amplitude envelope with controlled randomness
            # Number of control points for amplitude changes
            n_points = max(3, int(duration))
            
            # Generate random amplitude values within the specified range
            control_amplitudes = np.random.uniform(
                amplitude_range[0], 
                amplitude_range[1], 
                n_points
            )
            
            # Distribute control points across time
            control_times = np.linspace(0, duration, n_points)
            
            # Interpolate to create smooth amplitude changes (cubic spline)
            from scipy.interpolate import CubicSpline
            amplitude_envelope = CubicSpline(control_times, control_amplitudes)(time)
        else:
            # Static amplitude
            amplitude_envelope = np.ones(n_samples) * amplitude_range[0]
        
        # Generate base signal depending on signal_type
        if signal_type == "sine":
            signal_left = np.sin(2 * np.pi * frequency * time)
            signal_right = np.sin(2 * np.pi * frequency * time + phase_offset)
        elif signal_type == "square":
            signal_left = np.sign(np.sin(2 * np.pi * frequency * time))
            signal_right = np.sign(np.sin(2 * np.pi * frequency * time + phase_offset))
        elif signal_type == "sawtooth":
            from scipy import signal as scipy_signal
            signal_left = scipy_signal.sawtooth(2 * np.pi * frequency * time)
            signal_right = scipy_signal.sawtooth(2 * np.pi * frequency * time + phase_offset)
        elif signal_type == "noise":
            # White noise with specified correlation between channels
            signal_left = np.random.normal(0, 1, n_samples)
            
            # Control the correlation between channels
            correlation = np.cos(phase_offset)  # Map phase offset to correlation
            signal_right = correlation * signal_left + np.sqrt(1 - correlation**2) * np.random.normal(0, 1, n_samples)
        elif signal_type == "sweep":
            # Frequency sweep (useful for testing frequency-dependent behavior)
            from scipy import signal as scipy_signal
            f0 = frequency * 0.2  # Start frequency
            f1 = frequency * 5.0  # End frequency
            signal_left = scipy_signal.chirp(time, f0, duration, f1)
            signal_right = scipy_signal.chirp(time, f0, duration, f1, phi=phase_offset*180/np.pi)
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
        
        # Apply amplitude envelope
        signal_left *= amplitude_envelope
        signal_right *= amplitude_envelope
        
        # Apply fade in/out to avoid clicks
        if fade_edges:
            fade_samples = min(int(0.01 * self.SAMPLE_RATE), n_samples // 10)  # 10ms or 10% of signal
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            
            signal_left[:fade_samples] *= fade_in
            signal_left[-fade_samples:] *= fade_out
            signal_right[:fade_samples] *= fade_in
            signal_right[-fade_samples:] *= fade_out
        
        # Combine channels into stereo signal
        stereo_signal = np.vstack((signal_left, signal_right)).T
        
        # Ensure signal is in range [-1, 1]
        max_abs_value = np.max(np.abs(stereo_signal))
        if max_abs_value > 1.0:
            stereo_signal /= max_abs_value
        
        return stereo_signal

def parse_tuple(s):
    try:
        s = s.strip('()')
        return tuple(map(float, s.split(',')))
    except:
        raise argparse.ArgumentTypeError("Expected tuple format '(x,y)'")

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
        "-m", "--mode",
        help="Specify, if the signal should be complex or not",
        type=bool,
    )

    ap.add_argument(
        "-dr", "--dynamicRange",
        help="Set the dynamic range. In case of simple static amplitude use same values for lower and upper bound",
        type=parse_tuple,
        default=(-0.5,0.5)
    )
    
    args = ap.parse_args()

    generator = SignalGenerator(output_dir="./data/testing")
    
    frequency = args.frequency
    duration = args.duration
    n_run = args.run
    
    # parse dynamic range, keeping in mind, that bash only uses integer calculations
    dr = args.dynamicRange
    print(dr)
    phase_offset = SignalGenerator.MAX_PHASE_OFFSET / float(n_run)
    #convert radians to degree for clarity in stats
    deg = phase_offset * (180 / np.pi)
    #modulating freq: preferably low (<0.2) values for refined frequency variations
    mf = args.modFreq / 1000
    #modulation depth: higher values for stronger carrier modulation
    md = args.modDepth / 10
    name = args.name
    mode = args.mode
    #name = f"test_{frequency}Hz_{duration}s_{phase_offset}offset_{n_run}_run"
    print(bool(mode))

    result_dir = Path("./benchmark/results")
    # also taken care of in the batch script
    result_dir.mkdir(parents=True, exist_ok=True)

    sig, _ = generator.generate_signal(
        name=name,
        amplitude=dr,
        frequency=frequency,
        duration=duration,
        phase_offset=phase_offset,
        mod_frequency=mf,
        mod_depth=md,
        complex=bool(mode)
    )

    flattened_signal = np.ravel(a=sig,order="C")
    max_amp = np.amax(flattened_signal)
    min_amp = np.amin(flattened_signal)
    av_amp = np.mean(np.absolute(flattened_signal))
    dyn_r = max_amp - min_amp

    stats = {
        'frequency' : frequency,
        'duration' : duration,
        'phase_offset' : float(deg), # convert, because numpy float gave me some problems
        'modulating_frequency' : mf,
        'modulation_index' : md,
        'av_amplitude' : float(av_amp),
        'dynamic_range' : float(dyn_r),
        'singal_complexity' : mode,
    }

    json_path = result_dir / f"audio_test_{n_run}.json"
    with open(json_path, 'w') as f:  # Note the 'w' mode for writing
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    main(sys.argv[1:])