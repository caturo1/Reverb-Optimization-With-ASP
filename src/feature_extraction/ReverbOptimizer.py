import sys
from clingo.application import Application, clingo_main

class ReverbOptimizer(Application):
    def __init__(self, name):
        self.program_name = name
    
# TODO Implement reverbOptimizer Applicaiton