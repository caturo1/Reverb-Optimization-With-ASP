% our target parameters
% 0-1 mapped to 0-100 and vice versa
#const max = 95.
#const min = 5.
#const small_step = 5.
#const medium_step = 10.
#const large_step = 15.

size(R) :- R = min..max, R \ small_step = 0.
damping(D) :- D = min..max, D \ large_step = 0.
wet_level(W) :- W = min..100, W \ small_step = 0.
spread(S) :- S = 0..max, S \ medium_step = 0.

% our analyzed features and respective ranges
% features related to energy contained in the signal
% rms(0..100), 
% dr(0..100),
% density_population(0..10000), 
% features related to stereo information
% mid(0..100)
% side(0..100)
% rm_channel_balace(0..200),
% features related to spectral image
% spectral_centroid(0..20_000)