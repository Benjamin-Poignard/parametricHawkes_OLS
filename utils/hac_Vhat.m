function Vhat = hac_Vhat(nzIdx,nzVal,Tbins,Delta,beta,theta,mT,thetaWeight,useParallelForLags)

% Computes a HAC estimator of the long-run covariance matrix associated
% with the OLS score/moment vector for a discretized Hawkes regression.
%
% INPUTS:
%   nzIdx : vector containing the indices of the nonzero bins.
%           If nzIdx(j)=t, then bin t contains at least one event.
%   nzVal : vector containing the event counts in the corresponding
%           nonzero bins. Thus nzVal(j) is the number of events in
%           bin nzIdx(j).
%   Tbins : total number of bins in the estimation interval.
%   Delta : positive bin width used to discretize the Hawkes process.
%   beta  : K x 1 vector of exponential decay parameters beta_k.
%   theta : (K+1) x 1 parameter vector used to construct the regression
%           residuals/moment vectors, theta = [mu,alpha_1,...,alpha_K]
%           The length of theta must therefore equal 1+length(beta).
%   mT    : maximum lag included in the HAC estimator.
%           Covariances are accumulated for lags 0,...,mT.
%   thetaWeight :
%           nonnegative decay parameter controlling the HAC lag weights.
%           The weight assigned to lag ell is exp(-thetaWeight * ell),
%           for ell=0,...,mT.
%   useParallelForLags :
%           logical flag controlling whether the lag computations inside
%           the helper routines are carried out in parallel.
% OUTPUT:
%   Vhat  : (K+1) x (K+1) estimated HAC long-run covariance matrix.
%           The accumulated covariance matrix is normalized by Tbins
%           and symmetrized before being returned

beta = beta(:); theta = theta(:);
K = numel(beta); p = K + 1;

assert(numel(theta) == p, 'theta must have length 1 + length(beta).');

d = exp(-beta * Delta);
weights = exp(-thetaWeight * (0:mT));

mono = make_zero_monomial_table(beta, Delta, theta);

Vsum = zeros(p,p);
h = zeros(K,1);
gHist = zeros(0,p);
curr = 1;

for j = 1:numel(nzIdx)
    
    tEvt = nzIdx(j);
    ycount = nzVal(j);
    
    Lzero = tEvt - curr;
    
    if Lzero > 0
        [Vsum,h,gHist] = process_zero_run( ...
            Vsum,h,gHist,Lzero,beta,Delta,d,theta, ...
            mT,weights,mono,useParallelForLags);
    end
    
    [Vsum,h,gHist] = process_one_bin( ...
        Vsum,h,gHist,ycount,Delta,d,theta,mT,weights);
    
    curr = tEvt + 1;
end

Lzero = Tbins - curr + 1;

if Lzero > 0
    [Vsum,~,~] = process_zero_run( ...
        Vsum,h,gHist,Lzero,beta,Delta,d,theta, ...
        mT,weights,mono,useParallelForLags);
end

Vhat = Vsum / Tbins;
Vhat = (Vhat + Vhat.') / 2;