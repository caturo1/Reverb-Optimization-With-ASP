from clingo import Control, PropagateControl, PropagateInit, PropagatorCheckMode
from timeit import default_timer as timer
import sys
import os
from ..feature_extraction import ArtifactFeatures, load_audio, parameter_conversion
from ..audio import reverb_application as reverb

GLOBAL_READ = 0
GLOBAL_ANALYZE = 0
GLOBAL_REVERB = 0
GLOBAL_CHECKS = 0

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
    def __init__(self, display, input_name, input_path, input_features, output_dir, n_frames, dynamics, model_n):
        """
        Called once before solving to set up data structures used in theory propagation.

        Remark on artifact_thresholds:
            The artifact_features contain every analyzed feature seperately
            for the left and right channel, except for cross correlation.
            We will combine the channels in the check function.

        """

        self.__reassignments        = 0
        self.__input_feats          = input_features
        self.model_number           = model_n
        self.__artifact_thresholds  = {
            "clipping" : {
                "thresh" : 0.7,
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
                "thresh" : 1000,
                "count" : 4,
                "adjustment" : 1.1
            }
        }
        self.__input_path           = input_path
        self.__states               = {} ## Use a list to preserve states
        self.__symbols              = {}
        # set __display back to = display (just for benchmark True)
        self.__display              = True
        self.__dynamic              = dynamics
        # we need this to map solver literals back to parameter values
        self.__parameters           = {}
        self.__input_name           = input_name
        self.__output_dir           = output_dir

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
                self.__parameters[lit] = "selected_size"
                init.add_watch(lit)
            if atom.symbol.name == "selected_damp":
                lit = init.solver_literal(atom.literal)
                damp = int(str(atom.symbol.arguments[0]))
                self.__symbols[lit] = damp
                self.__parameters[lit] = "selected_damp"
                init.add_watch(lit)
            if atom.symbol.name == "selected_wet":
                lit = init.solver_literal(atom.literal)
                wet = int(str(atom.symbol.arguments[0]))
                self.__symbols[lit] = wet
                self.__parameters[lit] = "selected_wet"
                init.add_watch(lit)
            if atom.symbol.name == "selected_spread":
                lit = init.solver_literal(atom.literal)
                spread = int(str(atom.symbol.arguments[0]))
                self.__symbols[lit] = spread
                self.__parameters[lit] = "selected_spread"
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
        ## --> for cross correlation debug
        if self.__reassignments > 15 and self.__artifact_thresholds[conflict]["count"] > 0:
            if conflict == "cross-correlation":
                self.__artifact_thresholds[conflict]["thresh"] = \
                    (self.__artifact_thresholds[conflict]["thresh"][0] * self.__artifact_thresholds[conflict]["adjustment"],\
                        self.__artifact_thresholds[conflict]["thresh"][1] * self.__artifact_thresholds[conflict]["adjustment"])
                self.__artifact_thresholds[conflict]["count"] -= 1
                return
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

    def bulkcheck(self, artifact_features: ArtifactFeatures):
        if (artifact_features.b2mR_L > self.__artifact_thresholds["bass-to-mid"]["thresh"] or
            artifact_features.b2mR_R > self.__artifact_thresholds["bass-to-mid"]["thresh"] or
            artifact_features.clipping_r > self.__artifact_thresholds["clipping"]["thresh"] or
            artifact_features.clipping_l > self.__artifact_thresholds["clipping"]["thresh"] or
            artifact_features.cc < self.__artifact_thresholds["cross-correlation"]["thresh"][0] or 
            artifact_features.cc > self.__artifact_thresholds["cross-correlation"]["thresh"][1] or
            artifact_features.ringing_l > self.__artifact_thresholds["ringing"]["thresh"] or
            artifact_features.ringing_r > self.__artifact_thresholds["ringing"]["thresh"]):
            if (self.__display):
                print("Found artifacts.")
            return True
        
        return False
    
    def dynamic_check(self, artifact_features, nogood, state):
        """
        Depending on the violated artifact, add nogoods relevant to these artifacts.
        This is a try to minimize the backtracking in the CDCL algorithm.

        We will later compare the length of nogoods between this and bulkcheck,
        therefore just at literals if they aren't contained in the nogood list yet.
        """

        flag = False
        artifact = {
            'b2m': (artifact_features.b2mR_L > self.__artifact_thresholds["bass-to-mid"]["thresh"] or
                    artifact_features.b2mR_R > self.__artifact_thresholds["bass-to-mid"]["thresh"]),
            
            'cc': (artifact_features.cc < self.__artifact_thresholds["cross-correlation"]["thresh"][0] or
                artifact_features.cc > self.__artifact_thresholds["cross-correlation"]["thresh"][1]),
            
            'ringing': (artifact_features.ringing_l > self.__artifact_thresholds["ringing"]["thresh"] or
                    artifact_features.ringing_r > self.__artifact_thresholds["ringing"]["thresh"]),

            'clipping': (artifact_features.clipping_r > self.__artifact_thresholds["clipping"]["thresh"] or
                        artifact_features.clipping_l > self.__artifact_thresholds["clipping"]["thresh"])
        }

        def handle_artifacts(condition, nogood_params=None):
            """Helper function to handle common artifact processing logic"""
            nonlocal flag
            self.expansion(condition)
            self.__reassignments += 1
            
            for item, _ in state.items():
                if item not in nogood:
                    if nogood_params:
                        if self.__parameters[item] in nogood_params:
                            nogood.append(item)
                    else:
                        nogood.append(item)
            
            flag = True

        violated = next((key for key, is_violated in artifact.items() if is_violated), None)

        match violated:
            case 'b2m':
                handle_artifacts(
                    condition="bass-to-mid",
                    nogood_params=["selected_damp", "selected_size"]
                )
                
            case 'cc':
                handle_artifacts(
                    condition="cross-correlation",
                    nogood_params=["selected_wet", "selected_size", "selected_spread"]
                )
                
            case 'clipping':
                handle_artifacts(
                    condition="clipping",
                    nogood_params=["selected_wet", "selected_size"]
                )
                
            case 'ringing':
                handle_artifacts(
                    condition="ringing"
                )
                
        return flag


    def get_time_features():
        return GLOBAL_CHECKS, GLOBAL_ANALYZE, GLOBAL_READ, GLOBAL_REVERB
        

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
        
        #--> fix global issue, since if Optimizer runs for long sessions, they will accumulate over multiple instances
        global GLOBAL_READ
        global GLOBAL_ANALYZE
        global GLOBAL_CHECKS
        global GLOBAL_REVERB

        for lit, value in sorted(state.items()):
            ## 3) In the check function, apply the respective gains
            parameter_value = parameter_conversion(value)
            parameters[f"{self.__parameters[lit]}"] = parameter_value
            if (not self.__dynamic) :
                nogood.append(lit)

        ## 1) Apply reverb with the current parameters
        s10 = timer()

        # Create custom output file with version number according to number of models
        output_name = f"v{self.model_number}_processed_{self.__input_name}"
        output_path = os.path.join(self.__output_dir, output_name)
        reverb(
            input=self.__input_path,
            output=str(output_path),
            parameters=parameters)
        s11 = timer()
        el6 = s11 - s10
        GLOBAL_REVERB += el6

        ## 2) Load reverbated audio and run artifacts analyzer
        try:
            s4 = timer()
            output, _ = load_audio(output_path)
            s5 = timer()
            el3 = s5 - s4
            GLOBAL_READ += el3

            s6 = timer()
            artifact_features = ArtifactFeatures(
                y=output,
                mel_l_org=self.__input_feats.mel_left,
                mel_r_org=self.__input_feats.mel_right)
            s7 = timer()
            el4 = s7 - s6
            GLOBAL_ANALYZE += el4


        except Exception as e:
            print(f"Error {e} processing reverbrated audio.")
            sys.exit(1)
        

        s8 = timer()
        if (not self.__dynamic):
            res = self.bulkcheck(artifact_features)
            if (self.__display):
                print("Added bulk nogoods\n")
        else:
            res = self.dynamic_check(artifact_features, nogood, state)
            if (self.__display):
                print("Added only relevant nogoods\n")
        s9 = timer()
        el5 = s9 - s8
        GLOBAL_CHECKS += el5

        if display:
            print(f"Elapsed time for artifact checks: {el3}")
            print(f"Time for reapplying new reverb: {el6}")
            print(f"Time for reading new reverbrated output: {el3}")
            print(f"Time for analyzing new reverbated audio: {el4}\n\n")
            print(f"Assigned new parameters to {parameters} with artifact-check results:")
            print(f"Model number: {self.model_number}")
            artifact_features.to_string()

        if (res and
            (not control.add_nogood(nogood)
            or not control.propagate())):
                # reset relevant objects/files so they don't leak into the next run 
                if (self.__display):
                    print("Reset artifact_feature object and reverbrated audio "
                           "and explore different parts of the search space")
                del artifact_features
                if os.path.exists(str(output_path)):
                    os.remove(str(output_path))
                else:
                    print("Output file doesn't exist")
                    sys.exit(1)
                return

"""
        elif(artifact_features.den_diff_differential_l < self.__artifact_thresholds["density_difference"]["thresh"] or
            artifact_features.den_diff_differential_r < self.__artifact_thresholds["density_difference"]["thresh"] or
            artifact_features.den_stability_differential_l < self.__artifact_thresholds["density_stability"]["thresh"] or
            artifact_features.den_stability_differential_r < self.__artifact_thresholds["density_stability"]["thresh"]):
            self.expansion("density_difference")
            thresh = self.__artifact_thresholds["density_difference"]["thresh"]
            print(f"Is density_difference {artifact_features.den_diff_differential_l} or {artifact_features.den_diff_differential_l} < {thresh}")
            self.expansion("density_stability")
            self.__reassignments += 1
            if not control.add_nogood(nogood) or not control.propagate():
                return    
"""
