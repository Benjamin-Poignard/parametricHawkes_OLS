function out = bin_events(eventsTimes,Tmax,Delta)

% Binning for univariate Hawkes.
% INPUTS:
%   eventsTimes  : N x 1 event times
%   eventsMarks  : N x 1 marks (ignored except optional check; should be all 1)
%   Tmax         : horizon
%   Delta        : bin width
%
% OUTPUT:
%   out.Y        : T x 1 sparse count vector
%   out.t0       : T x 1 bin start times
%   out.Delta    : scalar
%   out.T        : number of bins

assert(Delta > 0, 'Delta must be > 0.');

T = ceil(Tmax / Delta);

mask = (eventsTimes >= 0) & (eventsTimes <= Tmax);
eventsTimes = eventsTimes(mask);

bin = floor(eventsTimes ./ Delta) + 1;
bin(bin > T) = T;

% counts per occupied bin only
[nzIdx, ~, ic] = unique(bin);
nzVal = accumarray(ic, 1);

out = struct();
out.T = T;             % scalar only
out.Delta = Delta;
out.nzIdx = nzIdx(:);  % occupied bin indices
out.nzVal = nzVal(:);  % counts in occupied bins