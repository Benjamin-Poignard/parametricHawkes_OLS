function Vblock = zero_run_hac_block(h0,lStart,lEnd,mT,weights,mono,p,useParallelForLags)

% Computes the HAC covariance contribution of a consecutive block of
% zero-count bins.
% The function evaluates the contribution of bins indexed from
% lStart to lEnd, assuming that no events occur in those bins.
% Because the Hawkes state decays deterministically during a zero run,
% the corresponding HAC terms can be accumulated analytically rather
% than processing every zero bin individually.
%
% INPUTS:
%   h0 : K x 1 Hawkes state vector at the beginning of the zero run.
%        Its k-th component represents the current excitation state
%        associated with kernel k.
%   lStart : first zero-bin index, relative to the beginning of the
%            zero-run representation used by the HAC routines.
%   lEnd   : last zero-bin index included in the block.
%            The number of zero bins processed is N = lEnd - lStart + 1.
%   mT     : maximum HAC lag. Contributions are accumulated for
%            tau = 0,...,mT.
%   weights : vector of HAC lag weights. The element weights(tau+1)
%             is the weight associated with lag tau.
%   mono    : structure containing precomputed quantities used to
%             evaluate the zero-run HAC terms analytically.
%             In this function, the following fields are used:
%             mono.thetaCoef, mono.hPowCount, mono.aRate
%             These encode the coefficients, powers of the initial
%             Hawkes state, and exponential decay rates entering the
%             monomial representation of the HAC terms.
%   p      : dimension of the moment/score vector. Consequently,
%            the returned HAC block has dimension p x p.
%   useParallelForLags :
%            logical flag. If true and mT >= 4, the contributions
%            corresponding to different HAC lags are computed using
%            parfor; otherwise they are computed sequentially.
%
% OUTPUT:
%   Vblock : p x p matrix containing the HAC covariance contribution
%            associated with the zero-count block lStart:lEnd,
%            summed over lags 0,...,mT.

Vblock = zeros(p,p);

if lEnd < lStart
    return;
end

N = lEnd - lStart + 1;

hPow = ones(numel(mono.thetaCoef),1);

for kk = 1:numel(h0)
    cnt = mono.hPowCount(:,kk);
    
    if h0(kk) == 0
        hPow(cnt > 0) = 0;
    else
        hPow = hPow .* (h0(kk) .^ cnt);
    end
end

geomPart = exp(-mono.aRate * lStart) .* geom_sum_exp_vec(mono.aRate,N);
baseVal = mono.thetaCoef .* hPow .* geomPart;

if useParallelForLags && mT >= 4
    Vcell = cell(mT+1,1);
    
    parfor tau0 = 0:mT
        Vcell{tau0+1} = zero_run_one_tau(baseVal,tau0,lStart,weights,mono,p);
    end
    
    for tau0 = 0:mT
        Vblock = Vblock + Vcell{tau0+1};
    end
else
    for tau0 = 0:mT
        Vblock = Vblock + zero_run_one_tau(baseVal,tau0,lStart,weights,mono,p);
    end
end