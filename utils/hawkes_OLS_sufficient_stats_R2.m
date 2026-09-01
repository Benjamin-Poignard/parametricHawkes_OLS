function [XtX, Xty, yTy, Sy] = hawkes_OLS_sufficient_stats_R2(nzIdx,nzVal,T,Delta,beta)

% Computes sufficient statistics for the discretized Hawkes OLS
% regression, including the quantities needed to compute R^2.
% The regression is y_t = mu + sum_{k=1}^K alpha_k h_{k,t} + error_t,
% where y_t = count in bin t / Delta, and
% h_{k,t} = sum_{ell<t} y_{ell*Delta} exp(-beta_k*Delta*(t-ell)).
% Only nonzero bins are supplied explicitly through nzIdx and nzVal.
% Consecutive zero-count bins are handled analytically.
%
% INPUTS:
%   nzIdx : vector containing the indices of the nonzero bins.
%           If nzIdx(j)=t, then bin t contains at least one event.
%   nzVal : vector containing the event counts in the corresponding
%           nonzero bins. Thus nzVal(j) is the number of events in
%           bin nzIdx(j).
%   T     : total number of bins in the estimation interval.
%   Delta : positive bin width used to discretize the Hawkes process.
%   beta  : K x 1 vector of positive exponential decay parameters.
%           These parameters are treated as fixed.
%
% OUTPUTS:
%   XtX   : (K+1) x (K+1) matrix equal to X'X, where x_t = [1; h_t] (regression-vector for bin t)
%   Xty   : (K+1) x 1 vector equal to X'y.
%   yTy   : scalar equal to y'y = sum_{t=1}^T y_t^2.
%   Sy    : scalar equal to sum_{t=1}^T y_t.
% These quantities can be used to compute the residual sum of squares
% and R^2 without storing the full regression matrix X or response y.

beta = beta(:);K = numel(beta);
nzIdx = nzIdx(:);nzVal = nzVal(:);
d = exp(-beta * Delta);

S1  = zeros(K,1);
S2  = zeros(K,K);
Sy  = 0;
Shy = zeros(K,1);
yTy = 0;

h = zeros(K,1);
curr = 1;

for j = 1:numel(nzIdx)
    
    tEvt = nzIdx(j);
    ycount = nzVal(j);
    
    % Zero run before occupied bin
    L = tEvt - curr;
    if L > 0
        [addS1, addS2, h] = accumulate_zero_run_R2(h, beta, Delta, L);
        S1 = S1 + addS1;
        S2 = S2 + addS2;
    end
    
    % Occupied bin
    yt = ycount / Delta;
    
    S1  = S1  + h;
    S2  = S2  + h*h.';
    Sy  = Sy  + yt;
    Shy = Shy + h*yt;
    yTy = yTy + yt^2;
    
    % Update state
    h = d .* h + ycount;
    
    curr = tEvt + 1;
end

% Trailing zero run
L = T - curr + 1;
if L > 0
    [addS1, addS2, ~] = accumulate_zero_run_R2(h, beta, Delta, L);
    S1 = S1 + addS1;
    S2 = S2 + addS2;
end

XtX = [T,   S1.';
    S1,  S2];

Xty = [Sy; Shy]; XtX = (XtX + XtX.') / 2;
