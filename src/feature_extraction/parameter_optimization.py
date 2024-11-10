import os
import sys
import numpy as np
from clingo import Model
from clingo.control import Control
from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile

import input_analysis as ia
import audio_features 

def write_instance(instance_file_path, instance) -> str:
    try: 
        with open(instance_file_path, 'r') as instance_file:
            instance_content = instance_file.read()
        new_instance_content = instance_content + instance
        
        try:
            with open(instance_file_path, 'w') as instance_file:
                instance_file.write(new_instance_content)
        except IOError:
            with open(instance_file_path, 'w') as instance_file:
                instance_file.write(instance_content)
            print("File not found. Restored instance.lp")
            raise
    except FileNotFoundError:
        print("Instance File not found")
        new_instance_content = instance
    return instance_content

def extract_params(model: Model) -> dict:
    params = {}
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

def check_mono(sample) -> int:
    return int(sample.ndim == 1) or int(sample.shape[0] == 1)


def main():
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    #sample = os.path.join(script_dir, '../../data/vocal[av_amp,clean,static,organic].wav')
    #sample = os.path.join(script_dir, '../../data/cello_climb[quiet,highMids,postprocessed,organic].wav')
    sample = os.path.join(script_dir, '../../data/bassline[loud,low_mid,synthetic,long].wav')
    #sample = os.path.join(script_dir, '../../data/short_spike[av_amp,mid-high,clean,synthetic].wav')
    asp_file_path = os.path.join(script_dir, '../ASP/encoding.lp')
    instance_file_path = os.path.join(script_dir, '../ASP/instance.lp')

    # extract values
    # TODO completely streamline features for stereo signals
    y, sr = ia.load_audio(sample)
    print(y.shape)
    S = ia.compute_STFT(y=y, sr=sr)

    # use sample as input for rms
    rms, rms_mean = ia.rms_features(y)
    print(rms.shape)
    dyn_rms = ia.compute_dynamic_rms(rms)
    mean_spectral_centroid = np.rint(ia.mean_spectral_centroid(y, sr))
    density = (100 - dyn_rms) * rms_mean
    s_flatness = ia.mean_spectral_flatness(y)
    s_rolloff = ia.compute_spectral_rolloff(y=y, sr=sr)
    np.set_printoptions(threshold=sys.maxsize)
    print(s_flatness.shape, s_rolloff.shape)
    mono = check_mono(y)

    # implement pipeline properly
    # maybe try snr with noise floor estimate as most silent part in the
    # recording

    # create current instance facts to parse into instance.lp
    instance = f"""
    rms({int(rms_mean)}).
    dr({int(dyn_rms)}).
    spectral_centroid({int(mean_spectral_centroid)}).
    density_population({int(density)}).
    mono({mono}).
    """

    base_content = write_instance(instance_file_path, instance)

    # start grounding and solve
    ctl = Control()
    ctl.load(instance_file_path)
    ctl.load(asp_file_path)
    #ctl.add("base", instance_file_path, asp_file_path)
    ctl.ground([("base", [])])
    #ctl.configuration.solve.models = 1
    
    params = {}

    print(f"ASP instance parameters: {instance}")

    with ctl.solve(yield_=True) as hnd:
        #extract reverb parameters here
        for model in hnd:
            #print(model)
            optimal_model = model
        params = extract_params(optimal_model)

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
#    output_file = os.path.splitext(sample)[0] + '_output.wav'

 
    with AudioFile(sample) as f:
        with AudioFile(output_file, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                audio = f.read(f.samplerate)
                effected = board(audio, f.samplerate, reset=False)
                o.write(effected)

    with open(instance_file_path, 'w') as f:
        f.write(base_content)


if __name__ == "__main__":
    main()
