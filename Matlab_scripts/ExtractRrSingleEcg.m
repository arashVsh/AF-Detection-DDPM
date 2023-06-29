function [QRS]=ExtractRrSingleEcg(x,sampleRate)

% input : raw signal
% output: r locations of baseline and noised removed signal
[X2] = baselinewanderremoval_BLACKSWAN(x); x = []; x = X2; X2 = [];
x  = x(:)'; x(1:2*sampleRate) = [];
[QRS, ~, ~] = qrs_detect21(x,0.25,0.6,sampleRate); 
