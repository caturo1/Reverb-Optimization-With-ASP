from clingo import Control, PropagateControl, PropagateInit, PropagatorCheckMode
import sys
from util import parameter_conversion
from ArtifactFeatures import ArtifactFeatures
import input_analysis as ia 
import reverb
import random

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
    def __init__(self, display, output_file_path, input_path, input_features, n_frames):
        """
        Called once before solving to set up data structures used in theory propagation.

        Remark on artifact_thresholds:
            The artifact_features contain every analyzed feature seperately
            for the left and right channel, except for cross correlation.
            We will combine the channels in the check function.

        """

        self.__output_path          = output_file_path
        self.__reassignments        = 0
        self.__input_feats          = input_features
        self.__artifact_features: ArtifactFeatures  = None
        self.__artifact_thresholds  = {
            "clipping" : {
                "thresh" : 0.4,
                "count" : 4,
                "adjustment" : 1.1
            },
            "bass-to-mid" : {
                "thresh" : 12,
                "count" : 4,
                "adjustment" : 1.1
            },
            "cross-correlation" : {
                "thresh" : (-0.3, 0.3),
                "count" : 3,
                "adjustment" : 1.1
            },
            "density_stability" : {
                "thresh" : 0.25,
                "count" : 5,
                "adjustment" : 0.8
            },
            "density_difference" : {
                "thresh" : 0.3 * n_frames,
                "count" : 5,
                "adjustment" : 1.25
            },
            "cluster_score" : {
                "thresh" : 50,
                "count" : 7,
                "adjustment" : 1.25
            },
            "ringing" : {
                "thresh" : 0.7,
                "count" : 4,
                "adjustment" : 1.1
            }
        }
        self.__input_path           = input_path
        self.__states               = {} ## Use a list to preserve states
        self.__symbols              = {}
        self.__display              = display
        self.__parameters           = {
            "selected_size" : 0,
            "selected_damp" : 0,
            "selected_wet" : 0,
            "selected_spread" : 0
        }
        
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

    def expansion(self, conflict: str) -> None:
        """
        Apply nogood and check, if we adjust thresholds.
        This is the core idea of having a dynamic artifact range, because subjectively speaking,
        some artifacts might add a some character we might like but we optimally want to avoid them.

        Remark: 
        - Idea 1) Dynamically adjust thresholds if we can't find a model that satisfies our artifact threholds. Speaking for this are the very individual feature scales and data types
        - Idea 2) Implement the thresholds in ASP encoding but search space then for first input analysis way larger (not yet implemented) and then use the specified artifact range as optimization statement

        Parameters:
        -----------
            conflict: The artifact we discovered, that violated our predefined thresholds
        """

        if self.__display:
                print(f"Oops, we have a {conflict} artifact! Add nogood")
        
        # if we added nogoods due to this specific artifact, relax the conditions a bit
        # but just a specified number of times
        # if we can still find no fitting model, than we are UNSAT
        if self.__reassignments > 30 and self.__artifact_thresholds[conflict]["count"] > 0:
            self.__artifact_thresholds[conflict]["thresh"] *= self.__artifact_thresholds[conflict]["adjustment"]
            self.__artifact_thresholds[conflict]["count"] -= 1

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
            ## !!! bug: not properly assigning identifiers to the parameters, therefore map solver literals to parameters
        for lit, value in sorted(state.items()):
            
            ## 3) In the check function, apply the respective gains
            parameter_value = parameter_conversion(value)
            parameters[f"{lit}"] = parameter_value

        if display:
            print(f"Assigned new parameters to {parameters}")

        ## 1) Apply reverb with the current parameters
        processed = reverb.reverb_application(
            input=self.__input_path,
            output=str(self.__output_path),
            parameters=parameters)

        ## 2) Load reverbated audio and run artifacts analyzer
        try:
            output, _ = ia.load_audio(processed)
        
        except Exception as e:
            print(f"Error {e} processing reverbrated audio")
            sys.exit(1)
        
        self.__artifact_features = ArtifactFeatures(
            y=output,
            mel_l_org=self.__input_feats.mel_left,
            mel_r_org=self.__input_feats.mel_right)
        
        if self.__display:
            self.__artifact_features.to_string()


        ## 3) Run Checks on Artifact features
        ## This can even be expanded to just adding nogoods that are relevant for the artifact
        if (self.__artifact_features.b2mR_L > self.__artifact_thresholds["bass-to-mid"]["thresh"] or 
            self.__artifact_features.b2mR_R > self.__artifact_thresholds["bass-to-mid"]["thresh"]):
            ## maybe damping or size or wet is too high
            self.expansion("bass-to-mid")
            self.__reassignments += 1
            relevant_nogoods = ["selected_damp", "selected_size"]
            
            for item in relevant_nogoods:
                nogood.append(self.__states[item])
            
            if not control.add_nogood(nogood) or not control.propagate():
                return    
            
        elif(self.__artifact_features.clipping_r > self.__artifact_thresholds["clipping"]["thresh"] or
             self.__artifact_features.clipping_l > self.__artifact_thresholds["clipping"]["thresh"]):
            self.expansion("clipping")
            self.__reassignments += 1
            relevant_nogoods = {"selected_wet", "selected_size"}
            
            for item in relevant_nogoods:
                nogood.append(self.__states[item])

            if not control.add_nogood(nogood) or not control.propagate():
                return    

        elif(self.__artifact_features.cc < self.__artifact_thresholds["cross-correlation"]["thresh"][0] or 
             self.__artifact_features.cc > self.__artifact_thresholds["cross-correlation"]["thresh"][1]):
            self.expansion("cross-correlation")
            print(self.__artifact_features.cc)
            self.__reassignments += 1
            relevant_nogoods = {"selected_spread", "selected_wet", "selected_size"}
            
            for item in relevant_nogoods:
                nogood.append(self.__states[item])

            if not control.add_nogood(nogood) or not control.propagate():
                return    
        
        elif(self.__artifact_features.clustering_differential_l > self.__artifact_thresholds["cluster_score"]["thresh"] or
            self.__artifact_features.clustering_differential_r > self.__artifact_thresholds["cluster_score"]["thresh"]):
            self.expansion("cluster_score")
            self.__reassignments += 1
            relevant_nogoods = {"selected_wet", "selected_size", "selected_spread", "selected_damp"}
            
            for item in relevant_nogoods:
                nogood.append(self.__states[item])

            if not control.add_nogood(nogood) or not control.propagate():
                return    

        elif(self.__artifact_features.ringing_l > self.__artifact_thresholds["ringing"]["thresh"] or
            self.__artifact_features.ringing_r > self.__artifact_thresholds["ringing"]["thresh"]):
            self.expansion("ringing")
            self.__reassignments += 1
            relevant_nogoods = {"selected_wet", "selected_size", "selected_spread", "selected_damp"}
            
            for item in relevant_nogoods:
                nogood.append(self.__states[self.__symbols[item]])

            if not control.add_nogood(nogood) or not control.propagate():
                return    

        else:
            pass

"""
        elif(self.__artifact_features.den_diff_differential_l < self.__artifact_thresholds["density_difference"]["thresh"] or
            self.__artifact_features.den_diff_differential_r < self.__artifact_thresholds["density_difference"]["thresh"] or
            self.__artifact_features.den_stability_differential_l < self.__artifact_thresholds["density_stability"]["thresh"] or
            self.__artifact_features.den_stability_differential_r < self.__artifact_thresholds["density_stability"]["thresh"]):
            self.expansion("density_difference")
            thresh = self.__artifact_thresholds["density_difference"]["thresh"]
            print(f"Is density_difference {self.__artifact_features.den_diff_differential_l} or {self.__artifact_features.den_diff_differential_l} < {thresh}")
            self.expansion("density_stability")
            self.__reassignments += 1
            if not control.add_nogood(nogood) or not control.propagate():
                return    
"""
