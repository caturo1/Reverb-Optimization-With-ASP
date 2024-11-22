import os
from clingo import Model
import time
from clingo.control import Control
from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile
import input_analysis as ia
from AudioFeatures import AudioFeatures
from AspHandler import AspHandler

def extract_params(model: Model, params: dict) -> dict:
    """Placeholder method for extracting model parameters"""
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
    return params

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
    ctl = Control()
    ctl.load(instance_file_path)
    ctl.load(asp_file_path)
    ctl.ground([("base", [])])
    
    params = {}

    with ctl.solve(yield_=True) as hnd:
        for model in hnd:
            mmodel = model
            print(model)
        params = extract_params(mmodel, params=params)

    print(params, type(params))
    
    # apply reverb
    output_dir = os.path.join(script_dir, '../../processed_data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'processed_file.wav')
    board = Pedalboard([Reverb(
                    room_size=params["size"],
                    damping=params["damping"],
                    wet_level=params["wet"],
                    dry_level=1 - params["wet"],
                    width=params["spread"]
                )])

    with AudioFile(sample) as f:
        with AudioFile(output_file, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                audio = f.read(f.samplerate)
                effected = board(audio, f.samplerate, reset=False)
                o.write(effected)


if __name__ == "__main__":
    main()
