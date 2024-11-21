from clingo import Model, Control
import AudioFeatures

class AspHandler:
    """
    A class that  handles ASP specific I/O

    Here we will hold relevant parameters to create the instance.lp
    for ASP guessing.
    """

    def __init__(self, instance_file_path, asp_file_path, features) -> None:
        self.instance_file_path = instance_file_path
        self.asp_file_path = asp_file_path
        self.input_instance = self.create_instance(features)
        self.base_content = self.write_instance()
        self.extract_params = self.extract_params()
        self.ctl = Control()


    def write_instance(self) -> str:
        """Writing the extracted input features to the instance.lp file
         
        In case of error: 
        - If we couldn't find the file return
        - If we opened the file, we'll save the contents for write back failures
        """

        try: 
            with open(self.instance_file_path, 'r') as instance_file:
                self.base_content = instance_file.read()
                new_base_content = self.base_content + self.instance
            
            try:
                with open(self.instance_file_path, 'w') as instance_file:
                    instance_file.write(new_base_content)
            except IOError:
                with open(self.instance_file_path, 'w') as instance_file:
                    instance_file.write(self.base_content)
                print("File not found. Restored instance.lp")
                raise
        except FileNotFoundError:
            print("Instance File not found")
        return self.base_content
    
    def create_instance(self, features: AudioFeatures):
        """Creation of an instance describing our input for ASP guessing"""
        
        self.input_instance = f"""
        rms({features.rms}).
        rms_left({features.rms_left}).
        rms_right({features.rms_right}).
        dr({features.dyn_rms}).
        density_population({features.density}).
        mid({features.mid}).
        side({features.side}).
        spectral_centroid({features.spectral_centroid}).
        spectral_flatness({features.spectral_flatness}).
        """
