import os
from clingo import Model
from clingo.control import Control
from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile
import input_analysis as ia
import feature_extraction.AudioFeatures as AudioFeatures 
import AspHandler

def main():
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample = os.path.join(script_dir, '../../data/vocal[av_amp,clean,static,organic].wav')
    asp_file_path = os.path.join(script_dir, '../ASP/encoding.lp')
    instance_file_path = os.path.join(script_dir, '../ASP/instance.lp')

    y, sr = ia.load_audio(sample)
    S = ia.compute_STFT(y=y, sr=sr)

    features = AudioFeatures(y, S)
    handle = AspHandler(instance_file_path, asp_file_path, features)

    ctl = Control()
    ctl.load(instance_file_path)
    ctl.load(asp_file_path)
    ctl.ground([("base", [])])
    
    params = {}

    with ctl.solve(yield_=True) as hnd:
        #extract reverb parameters here
        for model in hnd:
            #print(model)
            optimal_model = model
        params = extract_params(optimal_model)

    def extract_params(model: Model) -> dict:
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
        print(f"ASP instance parameters: {handle.instance}")
        with open(instance_file_path, 'w') as f:
            f.write(handle.base_content)

    
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
