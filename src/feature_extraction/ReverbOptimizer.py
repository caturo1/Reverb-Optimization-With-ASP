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
    version:str = "1.0"

    def __init__(self):
        self.ctl = Control()
        self.model = []
        self.input_file: Optional[str] = sys.argv[1]
        self.output_dir: str = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)), 
                '../../processed_audio')
        self.input_features: InputFeatures = None
        self.artifact_features: ArtifactFeatures = None

        os.makedirs(self.output_dir, exist_ok=True)

    def _on_model(self, model: Model, params: dict) -> None:
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

    def register_options(self, options: ApplicationOptions) -> None:
        """
        Register additional command-line options such as a string
        for audio input path, encoding path and instance path
        """

        group = "Reverb Options"
        options.add(
            group,
            option="input,i",
            description="audio file path",
            argument="<file>"
        )
        options.add(
            group,
            option="encoding,e",
            description="encoding file path",
            argument="<file>"
        )

    def main(self, files: Sequence[str]) -> None:
        """
        Main function implementing a multi-shot-solving attempt.

        Parameters:
        self: application object
        ctl: control handle
        files: list of files, namely the encoding and instances
        """

        if not files:
            files = ["-"]

        params = {}
        self.input_features = InputFeatures.read_input(self.input_file)
        files.append(self.input_features.instance_file_path)
        
        for file_ in files:
            self.ctl.load(file_)
        

        self.ctl.ground([("base",[])])
        solve_result = self.ctl.solve(on_model=self._on_model(params=params))
        
        if not solve_result.satisfiable:
            return

        processed = reverb.reverb_application(
            input=self.input_file, 
            output=self.output_dir, 
            parameters=params)
        
        self.artifact_features = ArtifactFeatures.read_output(processed)

        # not sure about being able to add instances like this to the running program
        self.ctl.add("artifact_facts", [], self.artifact_features.create_instance())
        self.ctl.ground([("artifact_facts", []), ("artifact", [])])
        self.ctl.solve(on_model=self._on_model(params=params))

        reverb.reverb_application(
            input=processed,
            output=self.output_dir,
            parameters=params
        )


def main():
    # the first argument is the audio input file path
    # the following argument is the encoding followed by 2 intance files?
    clingo_main(ReverbOptimizer(sys.argv[0]), sys.argv[2])