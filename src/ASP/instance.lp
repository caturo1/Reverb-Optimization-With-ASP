% our target parameters
% 0-1 mapped to 0-100 and vice versa
damping(0..100).
size(0..100).
spread(0..100).
wet_level(0..100).

% our analyzed features and respective ranges
% dr(0..100), rms(0..100), density_population(0..10000), 
% spectral_centroid(0..20000)
% We'll append the values dynamically here: