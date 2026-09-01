function [XtX,Xty,yTy,Sy] = hawkes_OLS_sufficient_stats_R2(nzIdx,nzVal,T,dt,beta)

% OLS-based loss function for Hawkes process
% Computes sufficient statistics for the discretized Hawkes OLS
% regression, useful in particular for evaluating the OLS loss and R^2.
% Only nonzero bins are supplied explicitly through nzIdx and nzVal.
% Zero-count stretches are handled analytically.
%
% INPUTS:
%   nzIdx : vector containing the indices of the nonzero bins.
%           If nzIdx(j)=t, then bin t contains at least one event.
%   nzVal : vector containing the event counts in the corresponding
%           nonzero bins. Thus nzVal(j) is the number of events in
%           bin nzIdx(j).
%   T     : total number of bins in the estimation interval.
%   dt    : positive bin width used to discretize the Hawkes process.
%   beta  : K x 1 vector of positive exponential decay parameters.
%           These parameters are treated as fixed in this routine.
%
% OUTPUTS:
%   XtX   : (K+1) x (K+1) matrix equal to X'X, where the first
%           regressor is the intercept and the remaining K regressors
%           are the Hawkes state variables h_{k,t}.
%   Xty   : (K+1) x 1 vector equal to X'y.
%   yTy   : scalar equal to y'y = sum_t y_t^2.
%   Sy    : scalar equal to sum_t y_t.

beta = beta(:);
K = numel(beta);
phi = exp(-beta*dt);

nzIdx = nzIdx(:);
nzVal = nzVal(:);

[nzIdx,~,ic] = unique(nzIdx);
nzVal = accumarray(ic,nzVal);

XtX = zeros(K+1,K+1);
Xty = zeros(K+1,1);
yTy = 0;
Sy  = 0;

h = zeros(K,1);
prev = 1;

for a = 1:numel(nzIdx)

    j = nzIdx(a);

    g = j - prev;
    if g > 0
        XtX = add_zero_block(XtX,h,phi,g);
        h = (phi.^g).*h;
    end

    x = [1; h];
    y = nzVal(a)/dt;

    XtX = XtX + x*x.';
    Xty = Xty + x*y;
    yTy = yTy + y^2;
    Sy  = Sy + y;

    h = phi.*h + nzVal(a);
    prev = j + 1;
end

g = T - prev + 1;
if g > 0
    XtX = add_zero_block(XtX,h,phi,g);
end
