function nll = HawkesMLE_full(theta,times,marks,Tmax,K)

% Negative log-likelihood for a univariate Hawkes process
% with K exponential kernels.
% Intensity: lambda(t) = mu + sum_{k=1}^K alpha_k sum_{t_i < t} exp(-beta_k*(t-t_i)).
%
% INPUTS:
%   theta : (1+2*K) x 1 parameter vector theta = [mu,alpha',beta']'
%           Here:
%             mu      = baseline intensity,
%             alpha_k = excitation coefficient of kernel k,
%             beta_k  = exponential decay parameter of kernel k.
%   times : N x 1 vector of observed event times.
%           The event times must be sorted in nondecreasing order,
%           satisfy times(i) >= 0, and lie in [0,Tmax].
%   marks : N x 1 vector of event marks.
%           Since the model is univariate, all marks must be equal
%           to one. Alternatively, marks may be empty [].
%   Tmax  : positive observation horizon. The likelihood is computed
%           over the interval [0,Tmax], and Tmax must satisfy
%           Tmax >= max(times).
%   K     : number of exponential kernels.
%
% OUTPUT:
%   nll   : scalar negative log-likelihood
%           nll = -ell(theta), where
%           ell(theta)= sum_{i=1}^N log(lambda_{t_i}(theta)) - integral_0^Tmax lambda_t(theta) dt.
%           If the parameter vector is invalid, the intensity becomes
%           nonpositive/nonfinite, or another numerical problem occurs,
%           the function returns the large penalty value 1e30.

if nargin < 5
    error('Need theta, times, marks, Tmax, K.');
end
if nargin < 4 || isempty(Tmax)
    Tmax = times(end);
end

times = times(:);
N = numel(times);

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

if ~isempty(marks)
    marks = marks(:);
    if numel(marks) ~= N
        error('marks must have same length as times (or be empty).');
    end
    if any(marks ~= 1)
        error('Univariate Hawkes: marks must all be 1 (or pass marks=[]).');
    end
end

if numel(theta) ~= 1 + 2*K
    error('theta must have length 1 + 2*K.');
end

mu    = theta(1);
alpha = theta(2:1+K);
beta  = theta(2+K:1+2*K);

if mu < 0 || any(alpha < 0) || any(beta <= 0) || any(~isfinite(theta))
    nll = 1e30;
    return;
end

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

    R = R + 1;
    tPrev = t;
end

dtT = Tmax - times.';
E = exp(-beta * dtT);
Sk = sum(1 - E, 2);

integral = mu * Tmax + sum((alpha ./ beta) .* Sk);

nll = -(logSum - integral);

if ~isfinite(nll)
    nll = 1e30;
end