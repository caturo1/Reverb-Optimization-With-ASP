#include "./instance.lp".

% only one value per parameter
1 { selected_size(R) : size(R) } 1.
1 { selected_damp(D) : damping(D) } 1.
1 { selected_wet(W) : wet_level(W) } 1.
1 { selected_spread(S) : spread(S) } 1.

% I think this might quickly lead to UNSAT
% Also rules are hard to set because the relationships aren't linear between reverb parameters and features
%selected_size(X) :- dr(D), X = 100 - D.


% Try to reduce excessive parameter setting that might create artifacts
% But we still want to hear the reverb and avoid models that don't apply reverb
% In this case we'd want UNSAT
:- selected_size(R), selected_wet(W), R + W > 170.
:- selected_wet(W), selected_damp(D), W > 60, D < 45.

% Set damp-related setting
:- selected_damp(D), spectral_centroid(C), C < 500, D > 25.
:- selected_damp(D), spectral_centroid(C), C > 4000, D < 50.
:- spectral_centroid(C), spectral_spread(P), selected_size(S), C < 600, P > 1000, S > 45.

:- spectral_centroid(C), 
   spectral_flatness(F),
   spectral_spread(S),
   selected_damp(D),
   selected_wet(W),
   C < 2000, F > 30, S > 1500, D < 40, W > 70.

% Set constraints if the sound is noisy
:- spectral_flatness(F), selected_damp(D), F > 75, D < 60.
:- spectral_flatness(F), selected_size(S), selected_wet(W), F > 70, S > 35, W > 50.

% Set constraints on energy (density, rms, rms difference between channels) across channels
:- density_population(P), selected_size(R), selected_wet(W), P < 400, R < 30, W < 20.
:- density_population(P), selected_size(R), P > 3500, R > 55.
%:- density_population(P), selected_damp(D), D > 80, P < 2500.
:- density_population(P), 
   selected_wet(W), 
   selected_damp(D), 
   P > 3500, W > 60, D < 40.

:- dr(D), selected_size(R), selected_wet(W), D < 30, R > 60, W > 25.
% conflict for 95 dynamic range, why is my wet only 5? This is very odd
:- dr(D), selected_size(R), selected_wet(W), D > 80, R < 20, W < 30.

:- rms_channel_balance(B), selected_spread(S), B > 60, S < 60.
:- rms_channel_balance(B), selected_spread(S), B < 20, S > 40, S < 10.
:- rms(E), selected_wet(W), selected_size(S), E > 90, W > 35, S > 50.
:- rms(E), selected_wet(W), E < 20, W < 60.

% Set spread-related parameters
% Roughly, if we have a mono/quite mono signal, we want higher spread parameters
% If we already have a spread input audio, the spread of the reverb should be small
:- mid(M), side(X), selected_spread(Y), X > M, Y > 30.
:- mid(M), side(X), selected_spread(Y), U = X + 10, M > U, Y < 40.
:- mid(M), side(X), selected_spread(Y), M + X > 80, Y > 60, Y < 15.
:- mid(M), side(X), selected_spread(Y), M + X < 10, Y < 40.

% weak constraints
%:~ selected_size(R), selected_wet(W), R > 80, W > 80. [1@1, R]
% if we have small size(room) and wet, we want to penalize wet  to at least hear sth
% :~ selected_size(R), selected_wet(W), W > 50. [1@1, R]

#show selected_size/1.
#show selected_damp/1.
#show selected_wet/1.
#show selected_spread/1.