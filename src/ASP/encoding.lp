% Rules for parameter selection
1 { selected_size(R) : size(R) } 1.
1 { selected_damp(D) : damping(D) } 1.
1 { selected_wet(W) : wet_level(W) } 1.
1 { selected_spread(S) : spread(S) } 1.

% Integrity constraints
:- density_population(P), selected_wet(W), selected_damp(SD), P > 7000, W > 40, SD < 40.
:- density_population(P), selected_wet(W), selected_damp(SD), P < 3000, W < 50, SD < 50.
:- selected_damp(D), spectral_centroid(S), S < 500, D < 25.
:- selected_damp(D), spectral_centroid(S), S > 10000, D > 50.
selected_spread(D) :- mono(D).

% scales don't align
% lets first choose the minimal parameter assignment as to avoid artifacts
% then maximize parameter (when artifact rules introduced) as to find optimal setting

#maximize { W@2 : selected_wet(W) }.
%, selected_size(S), selected_damp(SD), selected_spread(SSP)}.

% minimize artifacts with higher priority
% #minimize [ X@2 : someartifact(X) ]

#show selected_size/1.
#show selected_damp/1.
#show selected_spread/1. 
#show selected_wet/1.