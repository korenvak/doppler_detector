function y_out = AGNET(x, fs, params)
% AGNET Adaptive Gain and Noise Estimation Tool
%   y_out = AGNET(x, fs, params) applies a simple adaptive
%   gain control without resampling the signal.
%   This function preserves the original resolution of the
%   input vector x.

if nargin < 3
    params = struct();
end
alpha = getfield(params, 'alpha', 0.99);
max_gain = getfield(params, 'max_gain', 5.0);
noise_est = 0;
y_out = zeros(size(x));
for n = 1:length(x)
    noise_est = alpha * noise_est + (1-alpha) * abs(x(n));
    target = noise_est + eps;
    gain = min(max_gain, abs(x(n)) / target);
    y_out(n) = x(n) * gain;
end
end
