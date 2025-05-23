#include "./instance.lp".

% only one value per parameter
1 { selected_size(R) : size(R) } 1.
1 { selected_damp(D) : damping(D) } 1.
1 { selected_wet(W) : wet_level(W) } 1.
1 { selected_spread(S) : spread(S) } 1.

% Try to reduce excessive parameter setting that might create artifacts
% But we still want to hear the reverb and avoid models that don't apply reverb
:- selected_size(R), selected_wet(W), R + W > 170.
:- selected_wet(W), selected_damp(D), W > 60, D < 45.

% Set damp-related setting - for example, if the sound is too bright, we want to damp it and vice versa
:- spectral_centroid(C), selected_damp(D), C < 150, D > 25.
:- spectral_centroid(C), selected_damp(D), C > 4000, D < 50.
:- spectral_centroid(C), selected_damp(D), spectral_spread(S), C > 1000, S > 200, D < 30.
:- spectral_centroid(C), spectral_spread(P), selected_size(S), C < 600, P > 1000, S > 45.
:- spectral_centroid(C), 
   spectral_flatness(F),
   spectral_spread(S),
   selected_damp(D),
   selected_wet(W),
   C < 2000, F > 30, S > 250, D < 40, W > 70.

% Set constraints if the sound is noisy
:- spectral_flatness(F), selected_damp(D), F > 60, D < 60.
:- spectral_flatness(F), selected_size(S), selected_wet(W), F > 55, S > 35, W > 50.

% Set constraints on energy (density, rms, rms difference between channels) across channels
% ie., if the energy is too high, we want to reduce the size and wetness
:- density_population(P), selected_size(R), selected_wet(W), P < 400, R < 30, W < 20.
:- density_population(P), selected_size(R), P > 6000, R > 55.
:- density_population(P), 
   selected_wet(W), 
   selected_damp(D), 
   P > 5000, W > 60, D < 40.

:- dr(D), selected_size(R), selected_wet(W), D < 20, R > 60, W > 25.
:- dr(D), selected_size(R), D > 60, R < 25.
:- dr(D), selected_wet(W), D > 70, W < 35.

:- rms_channel_balance(B), selected_spread(S), B > 60, S < 60.
:- rms_channel_balance(B), selected_spread(S), B < 20 , S > 40, S < 10.
:- rms(E), selected_wet(W), selected_size(S), E > 90, W > 40, S > 50.
:- rms(E), selected_wet(W), E < 30, W < 50.

% Set spread-related parameters
% Roughly, if we have a mono/quite mono signal, we want higher spread parameters
% If we already have a spread input audio, the spread of the reverb should be small
:- mid(M), side(X), selected_spread(Y), X > M, Y > 30.
:- mid(M), side(X), selected_spread(Y), U = X + 10, M > U, Y < 40.
:- mid(M), side(X), selected_spread(Y), M + X > 80, Y > 60, Y < 15.
:- mid(M), side(X), selected_spread(Y), M + X < 10, Y < 40.

#show selected_size/1.
#show selected_damp/1.
#show selected_wet/1.
#show selected_spread/1.