% Rules for parameter selection
selected_size(R) : size(R).
selected_damp(D) : damping(D).
selected_wet(W) : wet_level(W).
selected_spread(S) : spread(S).

% Integrity constraints
% All the thresholds are purely experimental
:- density_population(P), selected_wet(W), selected_damp(SD), D > 7000, W > 40, SD < 40.
:- density_population(P), selected_wet(W), selected_damp(SD), D < 3000, W < 50, SD < 50.
:- selected_damp(D), spectral_centroid(S), S < 500, D < 25.
:- selected_damp(D), spectral_centroid(S), S > 10000, D > 50.
selected_spread() :- m 

% scales don't align
% lets first choose the minimal parameter assignment as to avoid artifacts
% then maximize parameter (when artifact rules introduced) as to find optimal setting
#maximize [ selected_wet(W)=W@1, selected_size(S)=S@1, selected_damp(SD)=SD@1, selected_spread(SSP)=70@1].
% minimize artifacts with higher priority
% #minimize [ someartifact=x@2 ]

#show selected_size/1 selected_damp/1 selected_spread/1 selected_wet/1