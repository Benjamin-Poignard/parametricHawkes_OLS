function est = hawkes_OLS(nzIdx,nzVal,T,Delta,beta)

% OLS estimator for Hawkes process (univariate) with K kernels
% Regression model: y_t / Delta = mu + sum_{k=1}^K alpha_k h_{k,t} + error_t,
% with h_{k,t} = sum_{ell < t} y_{ell*Delta} exp(-beta_k * Delta * (t-ell)).
% Only nonzero bins are passed explicitly through nzIdx and nzVal.
% Consecutive zero-count bins are handled analytically for efficiency.
%
% INPUTS:
%   nzIdx : vector containing the indices of the nonzero bins.
%           If nzIdx(j) = t, then the t-th bin contains at least
%           one event.
%   nzVal : vector containing the event counts in the corresponding
%           nonzero bins. Thus nzVal(j) is the number of events in
%           bin nzIdx(j).
%   T     : total number of bins in the estimation interval.
%           The corresponding continuous-time estimation horizon is
%           approximately T*Delta.
%   Delta : positive bin width used to discretize the Hawkes process.
%   beta  : K x 1 vector of strictly positive exponential decay
%           parameters beta_k. These parameters are treated as fixed
%           and known within this OLS estimation step.
%
% OUTPUT:
%   est : structure containing the OLS estimates and diagnostics:
%       est.mu
%           Estimated baseline intensity mu.
%       est.alpha
%           K x 1 vector of estimated excitation coefficients alpha_k.
%       est.theta
%           (K+1) x 1 parameter vector
%               [mu; alpha].
%       est.eta
%           Estimated total branching ratio
%               sum_k alpha_k / beta_k.
%       est.rho
%           K x 1 vector of component-wise branching ratios
%               alpha_k / beta_k.
%       est.theta_rho
%           (K+1) x 1 vector
%               [mu; alpha_1/beta_1; ...; alpha_K/beta_K].
%       est.XtX
%           (K+1) x (K+1) regression cross-product matrix X'X.
%       est.Xty
%           (K+1) x 1 regression cross-product vector X'y.
%       est.lambda
%           Ridge regularization parameter used for numerical
%           stabilization. Equal to zero unless X'X is detected
%           to be ill-conditioned.
%       est.eta_eps
%           Numerical tolerance used in the stationarity constraint
%               sum_k alpha_k/beta_k <= 1 - eta_eps.
%       est.exitflag
%           Exit flag returned by quadprog. The value -999 indicates
%           that quadprog failed and the fallback projected solution
%           was used.
%       est.output
%           Optimization information returned by quadprog.

assert(Delta > 0, 'Delta must be > 0.');
assert(T >= 1, 'T must be >= 1.');

beta = beta(:);
K = numel(beta);
assert(all(beta > 0), 'All beta_k must be > 0.');

nzIdx = nzIdx(:);
nzVal = nzVal(:);
assert(numel(nzIdx) == numel(nzVal), ...
    'nzIdx and nzVal must have same length.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Decay over one bin
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d = exp(-beta * Delta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sufficient statistics for regression:
% y_t = mu + sum_k alpha_k h_{k,t} + error_t
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S1  = zeros(K,1);
S2  = zeros(K,K);
Sy  = 0;
Shy = zeros(K,1);

h = zeros(K,1);
curr = 1;

for j = 1:numel(nzIdx)

    tEvt = nzIdx(j);
    ycount = nzVal(j);

    % Zero run before occupied bin
    L = tEvt - curr;
    if L > 0
        [addS1, addS2, h] = accumulate_zero_run(h, beta, Delta, L);
        S1 = S1 + addS1;
        S2 = S2 + addS2;
    end

    % Occupied bin
    yt = ycount / Delta;

    S1  = S1  + h;
    S2  = S2  + h*h.';
    Sy  = Sy  + yt;
    Shy = Shy + h*yt;

    % Update state
    h = d .* h + ycount;

    curr = tEvt + 1;
end

% Trailing zero run
L = T - curr + 1;
if L > 0
    [addS1, addS2, ~] = accumulate_zero_run(h, beta, Delta, L);
    S1 = S1 + addS1;
    S2 = S2 + addS2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Direct alpha-parametrization:
% theta = [mu; alpha]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
XtX = [T,   S1.';
    S1,  S2];

Xty = [Sy; Shy];

% Symmetrize for numerical stability
H = (XtX + XtX.') / 2;
f = -Xty;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Small ridge if ill-conditioned
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda = 0;

minEv = min(eig(H));
if minEv < 1e-10 || rcond(H) < 1e-12
    lambda = 1e-10 * max(mean(diag(H)), 1);
    D = diag([0; ones(K,1)]);   % do not penalize mu
    H = H + lambda * D;
    H = (H + H.') / 2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constraints:
% eps <= mu <= 100
% eps <= alpha_k <= max_k beta_k
% sum_k alpha_k / beta_k <= 1 - eta_eps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eta_eps = 1e-6; eps_pos = 1e-8;
mu_min = eps_pos; mu_max = 100;

lb = [mu_min; eps_pos*ones(K,1)];
ub = [mu_max; (max(beta)-eps_pos)*ones(K,1)];

Aineq = [0, (1./beta).'];
bineq = 1 - eta_eps;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Warm start: truncated unconstrained solution,
% projected onto the stationarity constraint if necessary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
theta0 = H \ Xty;
theta0 = min(max(theta0,lb),ub);

eta0 = sum(theta0(2:end)./beta);
if eta0 > 1-eta_eps
    theta0(2:end) = theta0(2:end) * ((1-eta_eps)/eta0);
end

opts = optimoptions('quadprog', ...
    'Display', 'off', ...
    'Algorithm', 'interior-point-convex');

[theta, ~, exitflag, output] = quadprog( ...
    H, f, Aineq, bineq, [], [], lb, ub, theta0, opts);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fallback if quadprog fails
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(theta)

    warning('quadprog failed in hawkes_OLS. Falling back to projected linear solve.');

    theta = H \ Xty;
    theta = min(max(theta,lb),ub);

    eta = sum(theta(2:end)./beta);
    if eta > 1-eta_eps
        theta(2:end) = theta(2:end) * ((1-eta_eps)/eta);
    end

    exitflag = -999;
    output = struct();
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Numerical cleanup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
theta = min(max(theta,lb),ub);

eta = sum(theta(2:end) ./ beta);
if eta > 1 - eta_eps
    theta(2:end) = theta(2:end) * ((1 - eta_eps) / eta);
    eta = sum(theta(2:end) ./ beta);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
est = struct();
est.mu        = theta(1);
est.alpha     = theta(2:end);
est.theta     = theta;
est.eta       = eta;
est.rho       = theta(2:end) ./ beta;
est.theta_rho = [theta(1); theta(2:end) ./ beta];

est.XtX       = XtX;
est.Xty       = Xty;
est.lambda    = lambda;
est.eta_eps   = eta_eps;
est.exitflag  = exitflag;
est.output    = output;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stable accumulation over L consecutive zero-count bins
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [addS1, addS2, hEnd] = accumulate_zero_run(h0, beta, Delta, L)

beta = beta(:);

% Sum_{ell=0}^{L-1} exp(-beta_k Delta ell)
a1 = beta * Delta;
g1 = geom_sum_exp(a1, L);

addS1 = h0 .* g1;

% Sum_{ell=0}^{L-1} exp(-(beta_i + beta_j) Delta ell)
a2 = (beta + beta.') * Delta;
G2 = geom_sum_exp(a2, L);

addS2 = (h0 * h0.') .* G2;

% Final state after L zero bins
hEnd = exp(-a1 * L) .* h0;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Numerically stable geometric sum:
% sum_{ell=0}^{L-1} exp(-a ell)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s = geom_sum_exp(a, L)

s = zeros(size(a));

small = abs(a) < 1e-14;

s(small) = L;

idx = ~small;
s(idx) = -expm1(-a(idx) * L) ./ (-expm1(-a(idx)));

end