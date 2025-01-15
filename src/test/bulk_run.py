
"""
Test script to run the ReverOptimizer application for every audio file located in 'data'.
"""

import os
import shutil
from sys import exit
from os import listdir
from clingo import clingo_main
from os.path import isfile, join
from ..application import ReverbOptimizer
from timeit import default_timer as timer
from contextlib import redirect_stdout, redirect_stderr


curr_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(curr_dir))
data_dir = os.path.join(grandparent_dir, "data")
argv = []
count = 0

onlyfile = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

def print_separator(text=""):
    terminal_width = shutil.get_terminal_size().columns
    print("\n")
    if text:
        text_with_spaces = f" {text} "
        dash_count = terminal_width - len(text_with_spaces)
        left_dashes = "-" * (dash_count // 2)
        right_dashes = "-" * (dash_count - len(left_dashes))
        print(f"{left_dashes}{text_with_spaces}{right_dashes}")
    else:
        print("-" * terminal_width)
    print("\n")

if not onlyfile:
    print(f"No audiofiles provided in {data_dir}")
    exit(1)

with open("output.txt", "w") as f:
     with redirect_stderr(f), redirect_stdout(f):   
        for file in onlyfile:
            argv = ["--audio-file", file, "--display", "--dynamic"]

            try:
                start = timer()
                res = clingo_main(ReverbOptimizer.ReverbOptimizer(), argv)
                if res == 10:
                    print("SATISFIABLE")
                else:
                    print("UNSATISFIABLE")
                end = timer()
            
            except Exception as e:
                print(f"Error processing {file}: {e}")
            
            print_separator(f"processed: {file} in {end-start:.2f} seconds")
            
            count += end - start

print(f"Finished testing with an overall runtime of {count:.2f} seconds")
