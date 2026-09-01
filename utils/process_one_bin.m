function [Vsum,hNext,gHist] = process_one_bin(Vsum,h,gHist,ycount,Delta,d,theta,mT,weights)

% Processes one discretized Hawkes bin and updates the HAC covariance
% accumulator, the Hawkes state, and the stored lagged moment vectors.
%
% INPUTS:
%   Vsum : current p x p accumulated HAC covariance matrix.
%   h    : K x 1 Hawkes state vector at the beginning of the bin.
%   gHist: stored moment vectors from previous bins, with at most
%          mT rows.
%   ycount: number of events observed in the current bin.
%   Delta: positive bin width.
%   d    : K x 1 one-bin decay vector, d_k = exp(-beta_k*Delta).
%   theta: p x 1 OLS parameter vector theta = [mu,alpha_1, ...,alpha_K].
%   mT   : maximum HAC lag.
%   weights: HAC lag weights, with weights(tau+1) corresponding
%            to lag tau.
%
% OUTPUTS:
%   Vsum : updated p x p HAC covariance accumulator.
%   hNext: K x 1 Hawkes state at the beginning of the next bin.
%   gHist: updated history of the most recent moment vectors.

x = [1; h];
u = ycount / Delta - theta.' * x; g = x * u;

Vsum = Vsum + weights(1) * (g * g.');

maxTau = min(mT, size(gHist,1));

if maxTau > 0
    for tau = 1:maxTau
        gLag = gHist(end - tau + 1,:).';
        Vsum = Vsum + weights(tau+1) * (g * gLag.' + gLag * g.');
    end
end

if mT > 0
    if size(gHist,1) < mT
        gHist = [gHist; g.'];
    else
        gHist(1:end-1,:) = gHist(2:end,:);
        gHist(end,:) = g.';
    end
end

hNext = d .* h + ycount;
