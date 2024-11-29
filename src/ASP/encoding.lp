#program base.

% only one value per parameter
1 { selected_size(R) : size(R) } 1.
1 { selected_damp(D) : damping(D) } 1.
1 { selected_wet(W) : wet_level(W) } 1.
1 { selected_spread(S) : spread(S) } 1.

:- selected_size(R), selected_wet(W), W + R > 130.
:- selected_size(R), R < 5.

:- selected_damp(D), spectral_centroid(S), S < 500, D > 25.
:- selected_damp(D), spectral_centroid(S), S > 4000, D < 50.
:- density_population(P), selected_damp(D), D > 80, P < 2500.
:- density_population(P), 
   selected_wet(SW), 
   selected_damp(SD), 
   P > 3500, SW > 40, SD < 40. 

:- spectral_centroid(C), 
   spectral_flatness(F), 
   spectral_spread(S),
   selected_damp(D),
   selected_wet(W),
   C < 2000, F > 30, S > 1500, D < 38, W > 70.

:- selected_wet(W), W < 5.
:- selected_wet(W), W > 95.
:- selected_size(R), R > 95.
:- selected_wet(W), selected_damp(D), W > 60, D < 46.

:- mid(M), side(X), selected_spread(Y), X > M, Y > 30.
:- mid(M), side(X), selected_spread(Y), M > X, Y < 40.

#show selected_size/1.
#show selected_damp/1.
#show selected_wet/1.
#show selected_spread/1.^


#program artifact.

