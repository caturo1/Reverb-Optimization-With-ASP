import sys
import os
import json
from timeit import default_timer as timer
from textwrap import dedent
from . import reverbPropagator as REVProp
from ..feature_extraction import InputFeatures, load_audio
from typing import Sequence
from clingo.application import Application, clingo_main, Flag
from clingo.control import Control

class ReverbOptimizer(Application):
    """
    Reverb Optimization Application.
    The application takes the name of the audio in question as input.

    Warning: The system is designed to only process clean (unrevebrated) audio

    Set "--dynamic" flag to dynamically insert artifact nogoods
    Set "--display" flag to get additional system information
    """
    
    program_name: str = "Reverb Optimization System"
    version: str = "1.0"

    def __init__(self):
        """
        Setup the data structures for the application.
        """
        self.__display : Flag                       = Flag(False)
        self.__audio_file                           = "data/"
        self.__encoding                             = "src/ASP/encoding.lp"
        self.__input_features: InputFeatures        = None
        self.answer_sets                            = []
        self.__dynamic                              = Flag(False)
        self.__base_content                         = ""
        # model number only changed here, so propagator works on the correct version
        self.__model_number                         = 1
        
        # initiate the directory for the processed audio
        self.output_dir: str = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)), 
                '..\..\processed_data')

    def dir_setup(self):
        """
        Set up directory for output files
        """

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            if self.__display:
                print("Output directory created.")
        else:
            if self.__display:
                print("Output directory already exists.")


    def __parse_audio_file(self, value):
        """
        Parse argument string
        """
        self.__audio_file = os.path.join(self.__audio_file, value)
        return True if isinstance(value, str) else False

    def register_options(self, options):
        """
        Extension point to add options to clingo-sp like choosing the
        transformation to apply.

        """
        group = "FX Processing Options"

        options.add(
            group, 
            "audio-file", 
            dedent("""Audio file to process. Default=demo.wav"""),
            self.__parse_audio_file)

        options.add_flag(
            group,
            "display",
            dedent("""Display more information"""),
            self.__display)
        
        options.add_flag(
            group,
            "dynamic",
            dedent("""Dynamically add nogoods depending on the violated artifact"""),
            self.__dynamic
        )

    def read_input(self, sample: str) -> None:
        """
        Read input and internally analyze features and create instance file
        
        Parameters:
        -----------
            sample: Path to the input audio
        """
        try: 
            y, sr = load_audio(sample)     
            
        except Exception as e:
            print(f"Error {e} processing input audio")
            sys.exit(1)

        self.__input_features = InputFeatures(y=y, sr=sr)
        self.store_base_content(self.__input_features.instance_file_path)
        inst = self.__input_features.create_instance()
        if self.__display:
            print(f"{inst}\n\n")

    def store_base_content(self, instance_file_path) -> None:
        if (not isinstance(instance_file_path, str)):
            print("Instance file path dubious")
            sys.exit(1)

        try:
            with open(instance_file_path) as f:
                    self.__base_content = f.read()
    
        except IOError as e:
            print(f"Input error {e}")
            sys.exit(1)
    
    def reset(self, instance_file_path):
        try:
            with open(instance_file_path, "r+") as f:
                f.seek(0)
                f.write(self.__base_content)
                f.truncate()
        except IOError as e:
            print("Couldn't reset instance.lp file")
            sys.exit(1)


    def main(self, ctl: Control, files: Sequence[str]) -> None:
        """
        Main function implementing a multi-shot-solving attempt.

        Parameters:
        self: application object
        ctl: control handle
        files: list of files, namely the encoding and instances
        """
        self.dir_setup()
        start = timer()
        ## Determine file path for reverbrated audio
        input_basename = os.path.basename(self.__audio_file)
        
        ## 1) Read input, analyze features, create instance
        if self.__display:
            print(f"Analyzing input audio {self.__audio_file}")
        s0 = timer()
        self.read_input(self.__audio_file)
        s1 = timer()
        el1 = s1 - s0
        
        ## 2) Load clingo encoding and input file
        if self.__display:
            print("Loading encodings")
        ctl.load(self.__encoding)
#        ctl.load(self.__input_features.instance_file_path)

        ## 3) Ground the encoding
        s2 = timer()
        if self.__display:
            print("Grounding...")
        ctl.ground([("base",[])])
        s3 = timer()
        el2 = s3 - s2

        ## 4) Register Propagator and solve according to its logic
        propagator = REVProp.reverbPropagator(display=self.__display,
                                        model_n=self.__model_number,
                                        input_name=input_basename,
                                        input_path=self.__audio_file,
                                        output_dir=self.output_dir,
                                        input_features=self.__input_features,
                                        n_frames=self.__input_features.mel_left.shape[1],
                                        dynamics=self.__dynamic)
        ctl.register_propagator(propagator)
        
        with ctl.solve(yield_=True) as hnd:
            
            for model in hnd:
                atoms_list = model.symbols(shown=True)
                self.answer_sets.append(atoms_list)
                print("\n SATISFIABLE \n")
                propagator.model_number += 1
            
            self.reset(self.__input_features.instance_file_path)

        checks_t, analyze_t, read_t, reverb_t = REVProp.reverbPropagator.get_time_features()

        
        end = timer()
        overall_t = end - start
        if self.__display:
            print(f"\n\nTime statistics: \n"
                  f"Input read and analyzation: {el1}\n"
                  f"Grounding: {el2}\n"
                  f"Reading reverbrated audio: {read_t}\n"
                  f"Analyzing reverbrated audio: {analyze_t}\n"
                  f"Checking for artifacts: {checks_t}\n"
                  f"Applying reverb: {reverb_t}\n"
                  f"Overall runtime: {overall_t}\n"
                  f"Solving time see clingo stats.")
        
        solving_choices = ctl.statistics['solving']['solvers']['choices']
        solving_conflicts = ctl.statistics['solving']['solvers']['conflicts']
        solving_rules = ctl.statistics['problem']['lp']['rules']
        constraints_stats = ctl.statistics['problem']['generator']['constraints']
        time_stats = ctl.statistics['summary']['times']
        # total time - solving time

        stats_output = {
            'choices': solving_choices,
            'conflicts': solving_conflicts,
            'constraints': constraints_stats,
            'time': time_stats,
            'rules': solving_rules
        }

        with open('stats.json', 'w') as f:
            json.dump(stats_output, f, indent=4)

    
if __name__ == "__main__":
    import warnings
    warnings.warn("use 'python -m application' not 'python -m application.ReverbOptimizer'", DeprecationWarning)
    sys.exit(int(clingo_main(ReverbOptimizer(), sys.argv[1:])))
