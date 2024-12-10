import sys
import os
import numpy as np
from InputFeatures import InputFeatures
from ArtifactFeatures import ArtifactFeatures
from ArtifactFeatures import *
from InputFeatures import *
from typing import Sequence, Optional
from clingo import ApplicationOptions, Model, Control
from clingo.application import Application, clingo_main
import reverb

class ReverbOptimizer(Application):
    """
    Reverb Optimization Application
    """
    
    program_name: str = "Reverb Optimization System"
    version: str = "1.0"

    def __init__(self):
#        self.model = []
        self.params = {}
        self.input_features: InputFeatures = None
        self.artifact_features: ArtifactFeatures = None
        
        # initiate the directory this application will save the processed audio to
        self.output_dir: str = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)), 
                '..\..\processed_data')

        os.makedirs(self.output_dir, exist_ok=True)

    def _on_model(self, model: Model) -> None:
        """Extract model parameters."""
        for symbol in model.symbols(shown=True):
            name = symbol.name
            value = symbol.arguments[0].number

            if name == "selected_size":
                self.params["size"] = value / 100
            elif name == "selected_damp":
                self.params["damping"] = value / 100
            elif name == "selected_wet":
                self.params["wet"] = value / 100
            elif name == "selected_spread":
                self.params["spread"] = value / 100

# kann ich auch zusammenlegen aber später
    def read_input(self, sample: str):
        """Read input and analyze features"""
        try: 
            y, sr = ia.load_audio(sample)     
            if not isinstance(y, np.ndarray):
                raise ValueError("Audio data must be of type numpy.ndarray")
            
        except Exception as e:
            print(f"Error {e} processing input audio")
            sys.exit(1)

        self.input_features = InputFeatures(y=y, sr=sr)
        self.input_features.create_instance()
    
    
    def read_output(self, sample: str):
        """Read processed audio and analyze features"""
        try: 
            output, sr = ia.load_audio(sample)

            if not isinstance(output, np.ndarray):
                raise ValueError("Audio data must be of type numpy.ndarray")
        
        except Exception as e:
            print(f"Error {e} processing input audio")
            sys.exit(1)

        self.artifact_features = ArtifactFeatures(
            y=output, 
            sr=sr,
            mel_l_org=self.input_features.mel_left, 
            mel_r_org=self.input_features.mel_right
            )


    def main(self, ctl, files: Sequence[str]) -> None:
        """
        Main function implementing a multi-shot-solving attempt.

        Parameters:
        self: application object
        ctl: control handle
        files: list of files, namely the encoding and instances
        """

        if not files:
            files = ["-"]

        input_file = files[0]
        asp_encoding = files[1]

        params = {}
        
        self.read_input(sample=input_file)

        instance_file = self.input_features.instance_file_path
        
        ctl.load(asp_encoding)
        ctl.load(instance_file)

        ctl.ground([("base",[])])
        solve_result = ctl.solve(on_model=self._on_model)
        
        if not solve_result.satisfiable:
            print("No model found")
            return
        
        input_basename = os.path.basename(input_file)
        output_filename = f"processed_{input_basename}"
        output_path = os.path.join(self.output_dir, output_filename)
        
        processed = reverb.reverb_application(
            input=input_file, 
            output=str(output_path), 
            parameters=self.params)
        
        self.read_output(sample=output_path)

        reverb.reverb_application(
            input=processed,
            output=self.output_dir,
            parameters=params
        )

if __name__ == "__main__":
    clingo_main(ReverbOptimizer())