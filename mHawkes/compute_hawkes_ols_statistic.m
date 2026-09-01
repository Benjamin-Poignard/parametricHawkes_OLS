function out = compute_hawkes_ols_statistic(eventsTimes,Tfinal,c,beta_mle,theta_true,mT,thetaWeight)

% Code for computation of the statistic
% Dhat_{T Delta}^{-1/2} * sqrt(T Delta) * (thetahat_{T Delta} - theta_true),
% where theta_hat = [mu_hat; alpha_hat] is obtained by OLS with beta fixed
% at beta_mle and Delta = c / max(beta_mle).
% This version does not rely on a dense Tbins-by-p design matrix. It uses
% the same occupied-bin representation as bin_events/hawkes_OLS, and computes
% the HAC matrix by exact zero-run accumulation.
%
% INPUTS:
%   eventsTimes   : event times in [0,Tfinal]
%   Tfinal        : final continuous-time horizon
%   c             : grid parameter, Delta = c/max(beta_mle)
%   beta_mle      : K x 1 beta vector; NOT max(beta)
%   theta_true    : (K+1) x 1 vector [mu_true; alpha_true]
%   mT            : HAC truncation lag, in number of bins
%   thetaWeight   : weights w_{T,tau}=exp(-thetaWeight*tau)
%   opts          : optional struct
%       .useParallelForLags : true/false, default false
%       .eigFloor          : eigenvalue floor multiplier, default 1e-10
%       .usePinv           : true/false, default true
%
% OUTPUTS:
%   out.z              : z statistic
%   out.theta_hat      : OLS estimator [mu_hat; alpha_hat]
%   out.Dhat           : estimated asymptotic variance matrix
%   out.Vhat           : HAC estimator of V
%   out.XtXoverT       : X'X/Tbins
%   out.Delta          : bin width
%   out.Tbins          : number of bins
%   out.TDelta         : Tbins*Delta
%   out.est            : output from hawkes_OLS
%
% REQUIREMENTS
%   hawkes_OLS.m must be on the MATLAB path and must return est.theta and
%   est.XtX in the direct alpha parametrization theta=[mu;alpha].

if nargin < 8 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'useParallelForLags'), opts.useParallelForLags = false; end
if ~isfield(opts, 'eigFloor'), opts.eigFloor = 1e-10; end
if ~isfield(opts, 'usePinv'), opts.usePinv = false; end
if ~isfield(opts, 'rcondTol'), opts.rcondTol = 1e-12; end

assert(c > 0, 'c must be positive.');
assert(Tfinal > 0, 'Tfinal must be positive.');
assert(mT >= 0 && floor(mT) == mT, 'mT must be a nonnegative integer.');
assert(thetaWeight >= 0, 'thetaWeight must be nonnegative.');

beta_mle = beta_mle(:);
K = numel(beta_mle);
p = K + 1;

theta_true = theta_true(:);
assert(numel(theta_true) == p, ...
    'theta_true must have length 1 + length(beta_mle).');
assert(all(beta_mle > 0), 'All beta_mle entries must be positive.');

Delta = c / max(beta_mle);

variable = bin_events_sparse(eventsTimes,Tfinal,Delta);

est = hawkes_OLS(variable.nzIdx,variable.nzVal,variable.T,Delta,beta_mle);
theta_hat = est.theta(:);

assert(numel(theta_hat) == p, ...
    'hawkes_OLS returned theta_hat with incompatible dimension.');

XtXoverT = est.XtX / variable.T;
XtXoverT = (XtXoverT + XtXoverT.') / 2;

Vhat = hac_Vhat(variable.nzIdx,variable.nzVal,variable.T, ...
    Delta,beta_mle,theta_hat,mT,thetaWeight,opts.useParallelForLags);

if opts.usePinv
    Ainv = pinv(XtXoverT);
else
    Ainv = safe_sym_solve_inverse(XtXoverT,opts.rcondTol);
end

Dhat = Ainv * Vhat * Ainv;
Dhat = (Dhat + Dhat.') / 2;

DhatInvHalf = sym_inv_sqrt(Dhat,opts.eigFloor);

TDelta = variable.T * Delta;
z = DhatInvHalf * (sqrt(TDelta) * (theta_hat - theta_true));

out = struct();
out.z = z;
out.theta_hat = theta_hat;
out.theta_true = theta_true;
out.Dhat = Dhat;
out.Vhat = Vhat;
out.XtXoverT = XtXoverT;
out.DhatInvHalf = DhatInvHalf;
out.Delta = Delta;
out.Tbins = variable.T;
out.TDelta = TDelta;
out.c = c;
out.beta_mle = beta_mle;
out.mT = mT;
out.thetaWeight = thetaWeight;
out.weights = exp(-thetaWeight * (0:mT));
out.est = est;
end

function Ainv = safe_sym_solve_inverse(A,rcondTol)

A = (A + A.') / 2;
p = size(A,1);

if rcond(A) > rcondTol
    Ainv = A \ eye(p);
else
    Ainv = pinv(A);
end

Ainv = (Ainv + Ainv.') / 2;
end

function out = bin_events_sparse(eventsTimes,Tmax,Delta)

assert(Delta > 0, 'Delta must be > 0.');
T = ceil(Tmax / Delta);

eventsTimes = eventsTimes(:);
eventsTimes = eventsTimes(eventsTimes >= 0 & eventsTimes <= Tmax);

if isempty(eventsTimes)
    nzIdx = zeros(0,1);
    nzVal = zeros(0,1);
else
    bin = floor(eventsTimes ./ Delta) + 1;
    bin(bin > T) = T;
    [nzIdx,~,ic] = unique(bin);
    nzVal = accumarray(ic,1);
end

out = struct('T',T,'Delta',Delta, ...
    'nzIdx',nzIdx(:),'nzVal',nzVal(:));
end
