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
GLOBAL_ARTIFACT_COUNT = 0


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
                "thresh" : 10,
                "count" : 4,
                "adjustment" : 1.1
            },
            "cross-correlation" : {
                "thresh" : (-0.3, 1.0),
                "count" : 3,
                "adjustment" : 1.1
            },
            "ringing" : {
                "thresh" : 1000,
                "count" : 4,
                "adjustment" : 1.1
            }
        }
        self.__input_path           = input_path
        # Dict to preserve states
        self.__states               = {} 
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

        Prameters
        ----------
            init : PropagateInit

        """
        # We need to set up the watches for the parameters we want to propagate
        for atom in init.symbolic_atoms:
            
            if atom.symbol.name == "selected_size":
                lit = init.solver_literal(atom.literal)
                size = int(str(atom.symbol.arguments[0]))
                # Maps literal to parameter value
                self.__symbols[lit] = size  
                # Maps literal to parameter name -> Alternative implementation with tuples as dict entries to reduce the number of lookups
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
        This is the core idea of having a dynamic artifact range, because
        defining a static threshold for a diverse set of inputs is hard and the perception of sound and artifacts to some degree is inherently subjective.

        After violating a threshold 15 times and not yet reassigning the artifact specific threshold a fixed number of times,
        we will adjust the threshold by an artifact specific factor.
        If the number of reassignments has been exhausted, we will not adjust the threshold anymore.

        Parameters:
            conflict: The artifact we discovered, that violated our predefined thresholds
        """

        if self.__display:
                print(f"We have a {conflict} artifact! Add nogood")
        
        # check number of overall and individual reassignments
        if self.__reassignments > 15 and self.__artifact_thresholds[conflict]["count"] > 0:
            # adjust the threshold and count
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
        Called whenever the solver retracts assignments
        to watched literals. It updates assignment dependent states in a propagator. 

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
        """
        Check for artifacts in the reverberated audio.

        Parameters:
            artifact_features: The artifact features object of the reverberated audio
        
        Returns:
            bool: True if artifacts are found, False otherwise
        """

        global GLOBAL_ARTIFACT_COUNT
        if (artifact_features.b2mR > self.__artifact_thresholds["bass-to-mid"]["thresh"] or
            artifact_features.clipping_r > self.__artifact_thresholds["clipping"]["thresh"] or
            artifact_features.clipping_l > self.__artifact_thresholds["clipping"]["thresh"] or
            artifact_features.cc < self.__artifact_thresholds["cross-correlation"]["thresh"][0] or 
            artifact_features.cc > self.__artifact_thresholds["cross-correlation"]["thresh"][1] or
            artifact_features.ringing_l > self.__artifact_thresholds["ringing"]["thresh"] or
            artifact_features.ringing_r > self.__artifact_thresholds["ringing"]["thresh"]):
            if (self.__display):
                print("Found artifacts.")
            GLOBAL_ARTIFACT_COUNT += 1
            return True
        
        return False
    
    def dynamic_check(self, artifact_features, nogood, state):
        """
        Depending on the violated artifact, add only relevant nogoods.

        Parameters:
            artifact_features: The artifact features object of the reverberated audio
            nogood: The nogood list
            state: The current state of the solver
        
        Returns:
            bool: True if artifacts are found, False otherwise
        """

        flag = False

        # build a dictionary with artifacts 
        artifact = {
            'b2m': (artifact_features.b2mR > self.__artifact_thresholds["bass-to-mid"]["thresh"]),
            
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
            global GLOBAL_ARTIFACT_COUNT

            self.expansion(condition)
            self.__reassignments += 1
            GLOBAL_ARTIFACT_COUNT = self.__reassignments
            
            # build nogoods
            for item, _ in state.items():
                if item not in nogood:
                    if nogood_params:
                        if self.__parameters[item] in nogood_params:
                            nogood.append(item)
                    else:
                        nogood.append(item)
            
            flag = True

        violated = next((key for key, is_violated in artifact.items() if is_violated), None)

        # switch cases for the violated artifacts
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
        """
        Returns the time statistics of the reverb propagator.
        """
        return GLOBAL_CHECKS, GLOBAL_ANALYZE, GLOBAL_READ, GLOBAL_REVERB, GLOBAL_ARTIFACT_COUNT
        

    def check(self, control: Control):
        """
        Check for artifacts in the reverberated audio and add nogoods if necessary.
        This function is called whenever the solver has a new assignment.
        """
        
        state       = self.__states
        display     = self.__display
        parameters = {}
        nogood   = []
        
        # collect stats over multiple runs
        global GLOBAL_READ
        global GLOBAL_ANALYZE
        global GLOBAL_CHECKS
        global GLOBAL_REVERB

        # extract assignment
        for lit, value in sorted(state.items()):
            parameter_value = parameter_conversion(value)
            parameters[f"{self.__parameters[lit]}"] = parameter_value
            if (not self.__dynamic) :
                nogood.append(lit)

        s10 = timer()

        # Apply reverb with the current parameters
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

        # Load reverbated audio and run artifacts analyzer
        try:
            s4 = timer()
            output, _ = load_audio(output_path)
            s5 = timer()
            el3 = s5 - s4
            GLOBAL_READ += el3

            s6 = timer()
            artifact_features = ArtifactFeatures(
                y_org=self.__input_feats.audio,
                y_proc=output,
                filter_bank_num_low=self.__input_feats.filterbank_low, 
                filter_bank_num_mid=self.__input_feats.filterbank_mid,
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
            # bulkcheck will check artifacts and add complete nogood
            res = self.bulkcheck(artifact_features)
            if (self.__display):
                print("Added bulk nogoods\n")
        else:
            # dynamic_check will check artifacts and add only relevant assignments
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

    	# if we found an artifact, we need to add a nogood and propagate
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