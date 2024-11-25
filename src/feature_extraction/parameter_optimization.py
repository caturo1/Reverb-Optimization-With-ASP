import os
import time
from clingo import Model, Control
from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile
import input_analysis as ia
from AudioFeatures import AudioFeatures
from AspHandler import AspHandler

def on_model(model: Model, params: dict) -> None:
    """Extract model parameters."""
    for symbol in model.symbols(shown=True):
        name = symbol.name
        value = symbol.arguments[0].number

        if name == "selected_size":
            params["size"] = value / 100
        elif name == "selected_damp":
            params["damping"] = value / 100
        elif name == "selected_wet":
            params["wet"] = value / 100
        elif name == "selected_spread":
            params["spread"] = value / 100

def run(instance_file_path: str, asp_file_path: str, params: dict) -> None:
    ctl = Control()
    ctl.load(instance_file_path)
    ctl.load(asp_file_path)
    ctl.ground([("base", [])])
    ctl.solve(on_model=lambda model: on_model(model, params))

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample = os.path.join(script_dir, '../../data/pure_noise[loud, high,noise,synthetic].wav')
    asp_file_path = os.path.join(script_dir, '../ASP/encoding.lp')
    instance_file_path = os.path.join(script_dir, '../ASP/instance.lp')

    y, sr = ia.load_audio(sample)
    S = ia.compute_STFT(y=y, sr=sr)

    start_time = time.perf_counter()

    features = AudioFeatures(y=y, sr=sr, S=S)

    end_time = time.perf_counter()
    time_elapsed = end_time - start_time
    file_duration = len(y[0]) / ia.RATE

    print(f"It took {time_elapsed:.2f} seconds for a {file_duration:.2f} second long file to initialize all the features, "
          f"which takes in relation {time_elapsed / file_duration:.2f} part of the signal length")

    AspHandler(instance_file_path, asp_file_path, features)

    # ASP guessing
    params = {}
    run(instance_file_path, asp_file_path, params)

    print(params, type(params))
    
    # Apply reverb
    output_dir = os.path.join(script_dir, '../../processed_data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'processed_file.wav')
    board = Pedalboard([Reverb(
                    room_size=params.get("size", 0.5),
                    damping=params.get("damping", 0.5),
                    wet_level=params.get("wet", 0.5),
                    dry_level=1 - params.get("wet", 0.5),
                    width=params.get("spread", 0.5)
                )])

    with AudioFile(sample) as f:
        with AudioFile(output_file, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                audio = f.read(f.samplerate)
                effected = board(audio, f.samplerate, reset=False)
                o.write(effected)

if __name__ == "__main__":
    main()
