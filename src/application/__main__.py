import sys
from clingo import clingo_main
import re
from . import ReverbOptimizer

def prechecks():
        pass

if __name__ == "__main__":
        
        print("Is the audio: \
              \n\t-not reverbrated \
              \n\t-not severely clipping?")
        
        answer = sys.stdin.readline()
        reg_exp = re.match(r'^y', answer, flags=re.IGNORECASE)
        if reg_exp is not None:
                print()
                prechecks()
                sys.exit(int(clingo_main(ReverbOptimizer.ReverbOptimizer(), sys.argv[1:])))
        else:
                print("Abort, please run RevOpt with proper input.")