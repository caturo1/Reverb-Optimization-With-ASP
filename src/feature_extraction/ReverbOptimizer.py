import sys
import os
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

    def __init__(self, string, input, encode):
#        self.model = []
        self.string = string
        self.params = {}
        self.input = input
        self.encoding = encode
        self.input_features: InputFeatures = None
        self.artifact_features = None
        
        # initiate the directory this application will save the processed audio to
        self.output_dir: str = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)), 
                '../../processed_audio')

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

    def register_options(self, options: ApplicationOptions) -> None:
        """
        Register additional command-line options so that we can
        pass an audio input path
        """

        group = "Reverb Options"
        options.add(
            group,
            option="input,i",
            parser=(str),
            description="audio file path",
            argument="<file>"
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
        
        self.input_features = InputFeatures.read_input(input_file)

        instance_file = self.input_features.instance_file_path
        
        ctl.load(asp_encoding)
        ctl.load(instance_file)

        ctl.ground([("base",[])])
        solve_result = ctl.solve(on_model=self._on_model)
        
        if not solve_result.satisfiable:
            return

        processed = reverb.reverb_application(
            input=self.input_file, 
            output=self.output_dir, 
            parameters=params)
        
        self.artifact_features = ArtifactFeatures.read_output(processed)

        # not sure about being able to add instances like this to the running program
        ctl.add(name="artifact_facts", parameters=[], program=self.artifact_features.create_instance())
        ctl.ground([("artifact_facts", []), ("artifact", [])])
        ctl.solve(on_model=self._on_model(params=self.params))

        reverb.reverb_application(
            input=processed,
            output=self.output_dir,
            parameters=params
        )


clingo_main(ReverbOptimizer(sys.argv[0], sys.argv[1], sys.argv[2]))

if __name__ == "__main__":
    clingo_main