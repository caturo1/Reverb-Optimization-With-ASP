import sys
from typing import Optional
from AudioFeatures import AudioFeatures

class AspHandler:
    """
    A class that  handles ASP specific I/O

    It holds relevant parameters to create the instance.lp
    for ASP guessing.
    """

    def __init__(self, instance_file_path, asp_file_path, features) -> None:
        self.instance_file_path = instance_file_path
        self.asp_file_path = asp_file_path
        self.input_instance = self.create_instance(features)
        self.write_instance()


    def write_instance(self) -> Optional[str]:
        """Append the extracted input features to the instance.lp file
         
        In case of error: 
        - We couldn't find the file return
        - We have a runtime IO error
        """
        if not self.input_instance:
            print("No instance to write")
            sys.exit(1)

        try:
            with open(self.instance_file_path, 'a') as instance_file:
                instance_file.write(self.input_instance)
        except FileNotFoundError:
            print(f"File {self.instance_file_path} not found.")
            sys.exit(1)
        except IOError as e:
            print(f" IO error when writing to {self.instance_file_path}")
            sys.exit(1)

    def create_instance(self, features: AudioFeatures) -> str:
        """Creation of an instance string describing our input for ASP guessing"""
        
        return f"""
    rms({int(features.rms_mean)}).
    rms_channel_balance({int(features.rms_channel_balance)}).
    dr({int(features.dynamic_range)}).
    density_population({int(features.density)}).
    mid({int(features.mid)}).
    side({int(features.side)}).
    spectral_centroid({int(features.spectral_centroid)}).
    spectral_flatness({int(features.spectral_flatness)}).
    spectral_spread({int(features.spectral_spread)})."""
