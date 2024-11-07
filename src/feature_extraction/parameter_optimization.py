import os
import numpy as np
from clingo import Model
import input_analysis as ia
from clingo.control import Control
import feature_extraction.audio_features as audio_features


def on_model(model: Model) -> None:
    # define what to do with the model:
    # we want to extract the parameters and apply reverb
    print(f"{model}")

def write_instance(instance_file_path, instance) -> None:
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


def main():
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample = os.path.join(script_dir, '../data/schreihals.wav')
    asp_file_path = (script_dir, '../ASP/encoding.lp')
    instance_file_path = (script_dir, '../ASP/instance.lp')
    
    # extract values
    y, sr = ia.load_audio(sample)
    S = ia.compute_STFT(y=y, sr=sr)
    
    # use sample as input for rms
    rms, rms_mean = ia.rms_features(y)
    dyn_rms = ia.compute_dynamic_rms(rms)
    mean_spectral_centroid = ia.mean_spectral_centroid(y, sr)
    density = (100 - dyn_rms) * rms_mean

    instance = f"""
    rms({rms_mean}).
    dr({dyn_rms}).
    spectral_centroid({mean_spectral_centroid}).
    density_population({density}).
    mono({0})
    """
    
    write_instance(instance_file_path, instance)

    ctl = Control()

    ctl.load(asp_file_path, instance_file_path)
    # maybe just compute one model
    ctl.gound([("base", [])])

    result = ctl.solve(on_model=on_model)

    print(result)