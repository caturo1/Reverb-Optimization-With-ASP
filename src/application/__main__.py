import sys
from clingo import clingo_main
from . import ReverbOptimizer

if __name__ == "__main__":
        sys.exit(int(clingo_main(ReverbOptimizer.ReverbOptimizer(), sys.argv[1:])))
