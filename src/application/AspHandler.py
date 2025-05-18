import sys

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

    @staticmethod
    def write_instance(input_instance, instance_file_path):
        """Append the extracted input features to the instance.lp file
         
        In case of error: 
        - We couldn't find the file return
        - We have a runtime IO error

        Parameters:
            input_instance : The input instance to be written to the file
            instance_file_path : The path to the instance file

        Returns:
            None
        """
        if not input_instance:
            print("No instance to write")
            sys.exit(1)

        try:
            with open(instance_file_path, 'a') as instance_file:
                instance_file.write(input_instance)
        except FileNotFoundError:
            print(f"File {instance_file_path} not found.")
            sys.exit(1)
        except IOError as e:
            print(f" IO error when writing to {instance_file_path}")
            sys.exit(1)


