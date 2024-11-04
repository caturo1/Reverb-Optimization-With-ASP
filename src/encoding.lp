% our target parameters
% 0-1 mapped to 0-100 and vice versa
damping(0..100).
room(0..100).
spread(0..100).
wet(0..100).

% our analyzed features
dr(0..100).
rms(0..100).
density_population(0..10000)

% Density correlation between inverse dynamic range and rms
inv_dynamic_range :- 100 - dr(DR)
density_population(D) :- rms(RMS), inv_dynamic_range(IDR), IDR * RMS

% Rules for parameter selection
{ selected_room(R) : room_size(R) } = 1.
{ selected_damp(D) : damping(D) } = 1.
{ selected_wet(W) : wet_level(W) } = 1.
{ selected_spread(S) : spread(S) } = 1.

% Optimization constraints
% All the thresholds are purely experimental
:- density_population(D), selected_wet(W), D > 80000, W > 40

% scales don't align
#minimize { |W - D| : selected_wet(W), density_population(D) }.