function nll = HawkesMLE(theta,times,marks,beta,Tmax)

% Negative log-likelihood for univariate Hawkes with multiple exponential kernels.
% Model: lambda(t) = mu + sum_{k=1}^K alpha_k * sum_{t_j < t} exp(-beta_k (t - t_j))
% theta = [mu; alpha_1; ...; alpha_K], with mu>=0, alpha_k>=0
% beta  = [beta_1; ...; beta_K], assumed known and beta_k>0
%
% INPUTS:
%   theta : (K+1)x1 vector [mu; alpha]
%   times : (N x 1) event times, sorted nondecreasing
%   marks : ignored (kept for compatibility); optionally all ones
%   beta  : K x 1 known decay rates
%   Tmax  : horizon; if empty uses times(end)
%
% OUTPUT:
%   nll   : negative log-likelihood for minimization

if nargin < 4
    error('Need theta, times, marks, beta.');
end
if nargin < 5 || isempty(Tmax)
    Tmax = times(end);
end

times = times(:);
beta = beta(:);
K = numel(beta);
N = numel(times);

if any(beta <= 0) || any(~isfinite(beta))
    nll = 1e30;
    return;
end

if N == 0
    mu = theta(1);
    if mu < 0 || ~isfinite(mu)
        nll = 1e30;
    else
        nll = mu * Tmax;
    end
    return;
end

assert(all(diff(times) >= 0), 'times must be sorted nondecreasing.');
assert(times(1) >= 0, 'times must be >= 0.');
assert(Tmax >= times(end), 'Tmax must be >= max(times).');

% optional sanity check marks
if ~isempty(marks)
    marks = marks(:);
    if numel(marks) ~= N
        error('marks must have same length as times (or be empty).');
    end
    if any(marks ~= 1)
        error('Univariate Hawkes: marks must all be 1 (or pass marks=[]).');
    end
end

% unpack parameters
mu = theta(1);
alpha = theta(2:end);

if numel(alpha) ~= K
    error('theta must have length K+1 = 1 + numel(beta).');
end

% parameter guards
if mu < 0 || any(alpha < 0) || any(~isfinite(theta))
    nll = 1e30;
    return;
end

% ----- 1) Sum log lambda(t_i^-) via vector recursion -----
% R_k(t) = sum_{t_j < t} exp(-beta_k (t - t_j))
R = zeros(K,1);
logSum = 0;
tPrev = 0;

for i = 1:N
    t = times(i);
    dt = t - tPrev;
    if dt < 0
        nll = 1e30;
        return;
    end

    if dt > 0
        R = R .* exp(-beta * dt);
    end

    lam = mu + alpha.' * R;
    if lam <= 0 || ~isfinite(lam)
        nll = 1e30;
        return;
    end
    logSum = logSum + log(lam);

    % jump: each kernel state gains +1 after the event
    R = R + 1;
    tPrev = t;
end

% ----- 2) Integral term -----
% ∫_0^T lambda(t) dt
% = mu*T + sum_k alpha_k * sum_j ∫_{t_j}^T exp(-beta_k (t-t_j)) dt
% = mu*T + sum_k (alpha_k / beta_k) * sum_j (1 - exp(-beta_k (T - t_j)))

dtT = Tmax - times.';   % 1 x N
E = exp(-beta * dtT);   % K x N
Sk = sum(1 - E, 2);     % K x 1

integral = mu * Tmax + sum((alpha ./ beta) .* Sk);

nll = -(logSum - integral);

if ~isfinite(nll)
    nll = 1e30;
end