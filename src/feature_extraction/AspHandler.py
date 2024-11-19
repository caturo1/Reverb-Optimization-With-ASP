from clingo import Model

class AspHandler:
    def __init__(self, instance_file_path, instance) -> None:
        self.instance_file_path = instance_file_path
        self.instance_content = instance

    def write_instance(self, instance: str) -> str:
            try: 
                with open(self.instance_file_path, 'r') as instance_file:
                    self.instance_content = instance_file.read()
                new_instance_content = self.instance_content + instance
                
                try:
                    with open(self.instance_file_path, 'w') as instance_file:
                        instance_file.write(new_instance_content)
                except IOError:
                    with open(self.instance_file_path, 'w') as instance_file:
                        instance_file.write(self.instance_content)
                    print("File not found. Restored instance.lp")
                    raise
            except FileNotFoundError:
                print("Instance File not found")
                new_instance_content = instance
            return self.instance_content

    @staticmethod
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