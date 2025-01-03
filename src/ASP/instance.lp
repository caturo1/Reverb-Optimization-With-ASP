% our target parameters
% 0-1 mapped to 0-100 and vice versa
#const max = 95.
#const min = 5.
#const small_step = 5.
#const medium_step = 10.
#const large_step = 15.

size(R) :- R = min..max, R \ large_step = 0.
damping(D) :- D = min..max, D \ medium_step = 0.
wet_level(W) :- W = min..100, W \ small_step = 0.
spread(S) :- S = 10..max, S \ medium_step = 0.

% our analyzed features and respective ranges
% features related to energy contained in the signal
% dr(0..100), 
% rms(0..100), 
% dr(0..100),
% density_population(0..10000), 
% features related to stereo information
% mid(0..100)
% side(0..100)
% cross_corr (0..200)
% rm_channel_balace(0..100),
% features related to spectral image
% spectral_centroid(0..20_000)
% spectral flatness ()

rms(26).
rms_channel_balance(0).
dr(77).
density_population(598).
mid(50).
side(0).
spectral_centroid(9207).
spectral_flatness(42).
spectral_spread(140).