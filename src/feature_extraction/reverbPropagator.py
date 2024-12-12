from clingo import Control, PropagateControl, PropagateInit, PropagatorCheckMode
from . import util
from ArtifactFeatures import *
import reverb

"""
The gist is:
For watching literals, we have to map:
symbolic_atoms -> program literals -> solver literals -> watches

How this propagator works
3)     In the check function, apply the respective gains
4)     Sum tracks to create the mix
5)     Create the periodogram and perform the corresponding reasoning module. In this case, look at the number of bands in range
6)     Minimize and perform lazy nogoods if needed
7)     If SAT, render plots and wav file
"""
    
class reverbPropagator:
    def __init__(self, display, output_file_path, input_path, input_features):
        """
        Called once before solving to set up data structures used in theory propagation.

        Remark on artifact_thresholds:
            The artifact_features contain every analyzed feature seperately
            for the left and right channel, except for cross correlation.
            We will combine the channels in the check function.

        """

        self.__output_path          = output_file_path
        self__no_good_set           = False
        self.__input_feats          = input_features
        self.__artifact_thresholds  = {
            "clipping" : 0.3,
            "bass-to-mid" : 12,
            "cross-correlation" : -0.3,
            "density_stability" : 0.25,
            "density_difference" : 0.25 * self.__artifact_features.mel_l.shape[1],
            "cluster_score" : 250,
            "ringing" : 1000
        } 
        self.__input_path           = input_path
        self.__states               = {} ## Use a list to preserve states
        self.__display              = display
        self.__symbols              = {}
        self.__artifact_features: ArtifactFeatures  = None

        
    def init(self, init: PropagateInit):
        """
        PropagateInit object to be handed to the init function.
        Gives us access to symbolic and theory atoms. Both are associated with
        program literals, that are in turn associated with solver literals.

        Here we set up watches used for propagation.
        Called once before each solving step.

        Prameters:
        ----------
            init: PropagateInit

        """

        ## No multithread support for now
        ## Fetch the parameter values and register them to watches
        for atom in init.symbolic_atoms:
            
            if atom.symbol.name == "selected_size":
                lit = init.solver_literal(atom.literal)
                size = int(str(atom.symbol.arguments[0]))
                self.__symbols[lit] = size
                init.add_watch(lit)
            if atom.symbol.name == "selected_damp":
                lit = init.solver_literal(atom.literal)
                damp = int(str(atom.symbol.arguments[0]))
                self.__symbols[lit] = damp
                init.add_watch(lit)
            if atom.symbol.name == "selected_wet":
                lit = init.solver_literal(atom.literal)
                wet = int(str(atom.symbol.arguments[0]))
                self.__symbols[lit] = wet
                init.add_watch(lit)
            if atom.symbol.name == "selected_spread":
                lit = init.solver_literal(atom.literal)
                spread = int(str(atom.symbol.arguments[0]))
                self.__symbols[lit] = spread
                init.add_watch(lit)
            
        init.check_mode = PropagatorCheckMode.Total

    def propagate(self, ctl: PropagateControl, changes):
        """
        Will be called during search to propagate solver literals, given a
        partial assignment.

        Parameters:
        -----------
            ctl: PropagateControl object,
                    which is used to inspect current assignment, record nogoods and trigger unit propagation.
                    Its threadID (seperate instance of CDCL) remarks currently active thread
            changes: Non-empty list of watched literals, that have been assigned to true w.r.t. current assignment
                    since the last call (in undo, propagate or solving_start)
        """
        state   = self.__states
        symbols = self.__symbols
        
        for lit in changes:
            state[lit] = symbols[lit]

    def undo(self, thread_id, assignment, changes):
        """
        Counterpart of propagate and called whenever the solver retracts assignments
        to watches literals. It updates assignment dependent states in a propagator
        but doesn't modify the current state of the solver. 
        This implements the backtracking in CDCL (I suppose)

        Parameters:
        -----------
            thread_id: Identifier of current thread
            assignment: Assignment of thread_id
            changes: List of watched literals
        """

        state   = self.__states
        
        for lit in changes:
            if lit in state:
                del state[lit]

    def check(self, control: Control):
        """
        Similar to propagate, yet invoked w/ changes and only called on total assignments.
        Independent of watched literals.        
        """

        ## 3)     In the check function, apply the respective FX
        ## 4)     Check for any artifacts
        ## 5)     If SAT, render new wav file, else, add nogood
        
        state       = self.__states
        display     = self.__display
        parameters = {}
        nogood   = []
        
        # now iterate over the state dictionary
        for lit, value in state.items():
            nogood.append(lit)

            ## 3) In the check function, apply the respective gains
            parameter_value = util.parameter_conversion(value)
            parameters[f"{lit}"] = parameter_value

        if display:
            print(f"Assigned new parameters to {parameters}\n"
                f"Will check reverbrated audio against artifact thresholds {self.__artifact_thresholds}")
            
        processed = reverb.reverb_application(
            input=self.__input_path, 
            output=str(self.__output_path), 
            parameters=parameters)

        self.__artifact_features = ArtifactFeatures(
            y=processed,
            mel_l_org=self.__input_feats.mel_left,
            mel_r_org=self.__input_feats.mel_right)
        
        if (self.__artifact_features.b2mR_L > self.__artifact_thresholds["bass-to-mid"] or 
            self.__artifact_features.b2mR_R > self.__artifact_thresholds["bass-to-mid"] or 
            self.__artifact_features.cc < self.__artifact_thresholds["clipping"] or
            self.__artifact_features.den_diff_differential_l < self.__artifact_thresholds["density_difference"] or
            self.__artifact_features.den_diff_differential_r < self.__artifact_thresholds["density_difference"] or
            self.__artifact_features.den_stability_differential_l < self.__artifact_thresholds["density_stability"] or
            self.__artifact_features.den_stability_differential_r < self.__artifact_thresholds["density_stability"] or
            self.__artifact_features.clustering_differential_l < self.__artifact_thresholds["cluster_score"] or
            self.__artifact_features.clustering_differential_r < self.__artifact_thresholds["cluster_score"] or
            self.__artifact_features.ringing_l < self.__artifact_thresholds["ringing"] or
            self.__artifact_features.ringing_r < self.__artifact_thresholds["ringing"]):
            if display and len(nogood)/4 < 30:
                print("conflict! add nogood")
            
            ## Idea 1) Dynamically adjust thresholds if we can't find a model that satisfies our artifact threhs
            ## Idea 2) Implement the thresholds in ASP encoding but search space then for first input analysis way larger
            if len(nogood) > 30:
                ## Possibly adjust parameters individually
                for key in self.__artifact_thresholds:
                    self.__artifact_thresholds[key] *= 0.7
            if not control.add_nogood(nogood) or not control.propagate():
                return
        else:
            pass
