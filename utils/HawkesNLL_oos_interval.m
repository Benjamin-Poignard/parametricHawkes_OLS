function nll = HawkesNLL_oos_interval(theta,trainTimes,testTimes,tStart,tEnd,K)

% Computes the out-of-sample negative log-likelihood of a univariate
% Hawkes process over a specified prediction interval.
% The likelihood is evaluated only over [tStart,tEnd), while conditioning
% on the event history observed before tStart.
%
% INPUTS:
%   theta      : (1+2*K) x 1 Hawkes parameter vector theta = [mu,alpha',beta']'
%                  mu      = baseline intensity,
%                  alpha_k = excitation coefficient of kernel k,
%                  beta_k  = exponential decay parameter of kernel k.
%   trainTimes : vector of historical event times observed before the
%                prediction interval. The function retains only events trainTimes < tStart
%                These events determine the Hawkes state at tStart
%   testTimes  : vector of events occurring in the out-of-sample
%                prediction period. The function retains only events tStart <= testTimes < tEnd.
%   tStart     : starting time of the out-of-sample evaluation interval.
%   tEnd       : ending time of the out-of-sample evaluation interval.
%                It must satisfy tEnd > tStart.
%   K          : number of exponential kernels.
%
% OUTPUT:
%   nll        : scalar out-of-sample negative log-likelihood over
%                [tStart,tEnd), nll = -ell_OOS(theta),
%                conditional on the event history before tStart.
%                If the parameters are invalid, an intensity is
%                nonpositive/nonfinite, or another numerical problem
%                occurs, the function returns 1e30.

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

trainTimes = trainTimes(:);
testTimes  = testTimes(:);

trainTimes = trainTimes(trainTimes < tStart);
testTimes  = testTimes(testTimes >= tStart & testTimes < tEnd);

if tEnd <= tStart
    nll = 1e30;
    return;
end

R = zeros(K,1);
hist = trainTimes(trainTimes < tStart);

for k = 1:K
    if isempty(hist)
        R(k) = 0;
    else
        R(k) = sum(exp(-beta(k) * (tStart - hist)));
    end
end

logSum = 0;
tPrev = tStart;

for i = 1:numel(testTimes)
    
    t = testTimes(i);
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

allPast = [trainTimes; testTimes];
allPast = allPast(allPast < tEnd);

integral = mu * (tEnd - tStart);

for k = 1:K
    tk = allPast(:);
    lower = max(tStart, tk);
    upper = tEnd;
    
    contrib = exp(-beta(k) * (lower - tk)) ...
        - exp(-beta(k) * (upper - tk));
    
    integral = integral + alpha(k) / beta(k) * sum(contrib);
end

nll = -(logSum - integral);

if ~isfinite(nll)
    nll = 1e30;
end