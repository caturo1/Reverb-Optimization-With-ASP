% our target parameters
% 0-1 mapped to 0-100 and vice versa
damping(0..100).
size(0..100).
spread(0..100).
wet_level(0..100).

% our analyzed features
dr(0..100).
rms(0..100).
density_population(0..10000).
spectral_centroid(0..20000).

% Rules for parameter selection
1 { selected_size(R) : size(R) } 1.
1 { selected_damp(D) : damping(D) } 1.
1 { selected_wet(W) : wet_level(W) } 1.
1 { selected_spread(S) : spread(S) } 1.

% Integrity constraints
% All the thresholds are purely experimental
:- density_population(P), selected_wet(W), selected_damp() D > 70000, W > 40.
:- selected_damp(D), spectral_centroid(S), S < 500, D < 25.

% scales don't align
% lets first choose the minimal parameter assignment as to avoid artifacts
% then maximize parameter (when artifact rules introduced) as to find optimal setting
#minimize { |W - D| : selected_wet(W)=100, density_population(D) }.


#show selected_size/1 selected_damp/1 selected_spread/1 selected_wet/1