% only one value per parameter
1 { selected_size(R) : size(R) } 1.
1 { selected_damp(D) : damping(D) } 1.
1 { selected_wet(W) : wet_level(W) } 1.
1 { selected_spread(S) : spread(S) } 1.

% Calc
%left(7).
%right(2).
%max(100).
%times (L * R) :- left(L), right(R).
%density(M - D) :- max(M), density_population(D).

% Integrity constraints
:- density_population(P), 
   selected_wet(W), 
   selected_damp(SD), 
   P > 3500, W > 40, SD < 40.

:- density_population(P), 
   selected_wet(W), 
   selected_damp(SD), 
   P < 3500, W < 50, SD < 50.

:- selected_damp(D), spectral_centroid(S), 
   S < 500, D > 25.

:- selected_damp(D), spectral_centroid(S), 
   S > 4000, D < 50.
selected_spread(D) :- mono(D).

:- selected_wet(W),
   selected_damp(D),
   W + D < 100.

:- selected_wet(W), W < 5, W > 95.
:- selected_size(R), R < 10.
:- selected_size(R), selected_wet(W), W + R > 150.

% scales don't necessarily align
% lets first choose the minimal parameter assignment as to avoid artifacts
% then maximize parameter (when artifact rules introduced) as to find optimal setting
#maximize { W@1 : selected_wet(W) }.
#minimize { X@2 : selected_damp(X) }.

% minimize artifacts with higher priority
% #minimize [ X@2 : someartifact(X) ]

#show selected_size/1.
#show selected_damp/1.
#show selected_spread/1. 
#show selected_wet/1.