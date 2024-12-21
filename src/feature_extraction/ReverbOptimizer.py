import sys
import os
import numpy as np
from textwrap import dedent
from reverbPropagator import reverbPropagator as REVProp
from ArtifactFeatures import ArtifactFeatures
from InputFeatures import InputFeatures
import input_analysis as ia
from typing import Sequence, cast
from reverb import reverb_application
from clingo.propagator import Propagator
from clingo.application import Application, clingo_main, Flag
from clingo.control import Control
from clingo.solving import SolveResult

class ReverbOptimizer(Application):
    """
    Reverb Optimization Application.
    The application takes the name of the audio in question as input.
    """
    
    program_name: str = "Reverb Optimization System"
    version: str = "1.0"

    def __init__(self):
        """
        Setup the data structures for the application.
        """
        self.__display : Flag                       = Flag(False)
        self.__audio_file                           = "../../data/"
        self.__encoding                             = "../ASP/encoding.lp"
        self.__input_features: InputFeatures        = None
        self.answer_sets                            = []
        self.__dynamic                              = Flag(False)
        self.__base_content                         = ""
        
        # initiate the directory for the processed audio
        self.output_dir: str = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)), 
                '..\..\processed_data')
        os.makedirs(self.output_dir, exist_ok=True)

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
        ----------
            sample: Path to the input audio
        """
        try: 
            y, sr = ia.load_audio(sample)     
            
        except Exception as e:
            print(f"Error {e} processing input audio")
            sys.exit(1)

        self.__input_features = InputFeatures(y=y, sr=sr)
        self.store_base_content(self.__input_features.instance_file_path)
        self.__input_features.create_instance()

    def store_base_content(self, instance_file_path) -> None:
        if (not isinstance(instance_file_path, str)):
            print("Instance file path dubious")
            sys.exit(1)

        try:
            with open(instance_file_path) as f:
                for line in f:
                    self.__base_content += line
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
        
        ## Determine file path for reverbrated audio
        input_basename = os.path.basename(self.__audio_file)
        output_filename = f"processed_{input_basename}"
        output_path = os.path.join(self.output_dir, output_filename)
        
        ## 1) Read input, analyze features, create instance
        if self.__display:
            print(f"Analyzing input audio {self.__audio_file}")
        self.read_input(self.__audio_file)
        
        ## 2) Load clingo encoding and input file
        if self.__display:
            print("Loading encodings")
        ctl.load(self.__encoding)
#        ctl.load(self.__input_features.instance_file_path)

        ## 3) Ground the encoding
        if self.__display:
            print("Grounding...")
        ctl.ground([("base",[])])

        ## 4) Register Propagator and solve according to its logic
        ctl.register_propagator(REVProp(display=self.__display,
                                        output_file_path=output_path,
                                        input_path=self.__audio_file,
                                        input_features=self.__input_features,
                                        n_frames=self.__input_features.mel_left.shape[1],
                                        dynamics=self.__dynamic))
        
        with ctl.solve(yield_=True) as hnd:
            for model in hnd:
                atoms_list = model.symbols(shown=True)
                self.answer_sets.append(atoms_list)
                if self.__display:
                    print()

        self.reset(self.__input_features.instance_file_path)

if __name__ == "__main__":
    sys.exit(int(clingo_main(ReverbOptimizer(), sys.argv[1:])))
