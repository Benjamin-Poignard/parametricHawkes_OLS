function predCount = hawkes_predict_count_stationary(horizonSec,mu,alpha,beta)

% Count prediction based on stationary formula

alpha = alpha(:);
beta  = beta(:);

eta = sum(alpha ./ beta);

if mu < 0 || eta >= 1 || any(alpha < 0) || any(beta <= 0)
    predCount = NaN;
    return;
end

lambdaBar = mu / (1 - eta);
predCount = horizonSec * lambdaBar;

if ~isfinite(predCount) || predCount < 0
    predCount = NaN;
end