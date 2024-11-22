% our target parameters
% 0-1 mapped to 0-100 and vice versa
damping(0..100).
size(0..100).
spread(0..100).
wet_level(0..100).

% our analyzed features and respective ranges
% features related to energy contained in the signal
% dr(0..100), 
% rms(0..100), 
% dr(0..100),
% density_population(0..10000), 
% features related to stereo information
% mid(0..100)
% side(0..100)
% rm_channel_balace(0..100),
% features related to spectral image
% spectral_centroid(0..20_000)
% spectral flatness ()
% spectral_spread(0.20_000)
% We'll append the values dynamically here:

rms(20).
rms_channel_balance(0).
dr(83).
density_population(340).
mid(50).
side(49).
spectral_centroid(910).
spectral_flatness(3).
spectral_spread(1205).
rms(71).
rms_channel_balance(10).
dr(66).
density_population(2414).
mid(51).
side(43).
spectral_centroid(1145).
spectral_flatness(0).
spectral_spread(1630).