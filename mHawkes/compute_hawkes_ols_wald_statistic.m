function out = compute_hawkes_ols_wald_statistic(eventsTimes,Tfinal,c,beta_mle,R,r,mT,thetaWeight,opts)

% Code for computation of the Wald statistic:
%   S_{T Delta} = T * (R*thetahat_{T Delta} - r)' * (R*Dhat_{T Delta}*R')^{-1} * (R*thetahat_{T Delta} - r).
% Here theta_hat = [mu_hat; alpha_hat] is obtained by OLS with beta fixed
% at beta_mle and Delta = c / max(beta_mle). The beta terms are NOT included
% in theta_hat.
%
% This version does not rely on a dense Tbins-by-p design matrix. It uses
% the same occupied-bin representation as bin_events/hawkes_OLS, and computes
% the HAC matrix by exact zero-run accumulation, as in
% compute_hawkes_ols_statistic.
%
% INPUTS:
%   eventsTimes   : event times in [0,Tfinal]
%   Tfinal        : final continuous-time horizon
%   c             : grid parameter, Delta = c/max(beta_mle)
%   beta_mle      : K x 1 beta vector; NOT max(beta)
%   R             : q x (K+1) restriction matrix
%   r             : q x 1 restriction vector
%                   Null hypothesis: R*theta = r
%   mT            : HAC truncation lag, in number of bins
%   thetaWeight   : weights w_{T,tau}=exp(-thetaWeight*tau)
%   opts          : optional struct
%       .useParallelForLags : true/false, default false
%       .eigFloor          : eigenvalue floor multiplier, default 1e-10
%       .usePinv           : true/false, default true
%       .waldScale         : 'Tbins' or 'TDelta', default 'Tbins'
%
% OUTPUTS:
%   out.S              : Wald statistic using opts.waldScale
%   out.S_Tbins        : Wald statistic with multiplier Tbins
%   out.S_TDelta       : Wald statistic with multiplier Tbins*Delta
%   out.q              : number of restrictions
%   out.theta_hat      : OLS estimator [mu_hat; alpha_hat]
%   out.Rtheta_minus_r : R*theta_hat - r
%   out.RDhatR         : R*Dhat*R'
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

if nargin < 9 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'useParallelForLags'), opts.useParallelForLags = false; end
if ~isfield(opts, 'waldScale'), opts.waldScale = 'Tbins'; end

assert(c > 0, 'c must be positive.');
assert(Tfinal > 0, 'Tfinal must be positive.');
assert(mT >= 0 && floor(mT) == mT, 'mT must be a nonnegative integer.');
assert(thetaWeight >= 0, 'thetaWeight must be nonnegative.');

beta_mle = beta_mle(:);
K = numel(beta_mle);
p = K + 1;

assert(all(beta_mle > 0), 'All beta_mle entries must be positive.');

R = double(R);
r = r(:);

assert(size(R,2) == p, ...
    'R must have 1 + length(beta_mle) columns.');
assert(size(R,1) == numel(r), ...
    'r must have the same number of rows as R.');
assert(size(R,1) >= 1, ...
    'R must contain at least one restriction.');

q = size(R,1);

Delta = c / max(beta_mle);

variable = bin_events_sparse(eventsTimes, Tfinal, Delta);

est = hawkes_OLS(variable.nzIdx, variable.nzVal, variable.T, Delta, beta_mle);
theta_hat = est.theta(:);

assert(numel(theta_hat) == p, ...
    'hawkes_OLS returned theta_hat with incompatible dimension.');

XtXoverT = est.XtX / variable.T;
XtXoverT = (XtXoverT + XtXoverT.') / 2;

Vhat = hac_Vhat(variable.nzIdx, variable.nzVal, variable.T, ...
    Delta, beta_mle, theta_hat, mT, thetaWeight, opts.useParallelForLags);

Ainv = XtXoverT \ eye(p);

Dhat = Ainv * Vhat * Ainv;
Dhat = (Dhat + Dhat.') / 2;

d = R * theta_hat - r;

RDhatR = R * Dhat * R.';
RDhatR = (RDhatR + RDhatR.') / 2;

if q == 1
    RDhatRinv = 1 / RDhatR;
else
    RDhatRinv = RDhatR \ eye(q);
end

Tbins = variable.T;
TDelta = Tbins * Delta;

quad = d.' * RDhatRinv * d;
quad = real(quad);

S_Tbins  = Tbins  * quad;
S_TDelta = TDelta * quad;

switch lower(opts.waldScale)
    case {'tbins','t','eq22'}
        S = S_Tbins;
        scaleUsed = 'Tbins';
    case {'tdelta','t_delta','td'}
        S = S_TDelta;
        scaleUsed = 'TDelta';
    otherwise
        error('Unknown opts.waldScale. Use ''Tbins'' or ''TDelta''.');
end

out = struct();
out.S = S;
out.S_Tbins = S_Tbins;
out.S_TDelta = S_TDelta;
out.scaleUsed = scaleUsed;
out.q = q;
out.theta_hat = theta_hat;
out.R = R;
out.r = r;
out.Rtheta_minus_r = d;
out.RDhatR = RDhatR;
out.RDhatRinv = RDhatRinv;
out.Dhat = Dhat;
out.Vhat = Vhat;
out.XtXoverT = XtXoverT;
out.Delta = Delta;
out.Tbins = Tbins;
out.TDelta = TDelta;
out.c = c;
out.beta_mle = beta_mle;
out.mT = mT;
out.thetaWeight = thetaWeight;
out.weights = exp(-thetaWeight * (0:mT));
out.est = est;
end

function out = bin_events_sparse(eventsTimes, Tmax, Delta)

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
    
    [nzIdx, ~, ic] = unique(bin);
    nzVal = accumarray(ic, 1);
end

out = struct();
out.T = T;
out.Delta = Delta;
out.nzIdx = nzIdx(:);
out.nzVal = nzVal(:);
end