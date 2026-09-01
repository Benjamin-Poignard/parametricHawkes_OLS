function loss = hawkes_OLS_insample_loss(nzIdx,nzVal,T,Delta,beta,theta)

% Computes the same in-sample residual MSE associated with the binned
% OLS regression:
%   y_t = count_t / Delta
%   yhat_t = mu + alpha' h_t
%   loss = (1/T) sum_t (y_t - yhat_t)^2
%
% using sparse occupied-bin representation.

beta = beta(:);
theta = theta(:);

K = numel(beta);
mu = theta(1);
alpha = theta(2:end);

d = exp(-beta * Delta);

nzIdx = nzIdx(:);
nzVal = nzVal(:);

h = zeros(K,1);
curr = 1;

SSE = 0;

for j = 1:numel(nzIdx)
    
    tEvt = nzIdx(j);
    ycount = nzVal(j);
    
    % Zero run before occupied bin
    L = tEvt - curr;
    if L > 0
        [s1, s2] = zero_run_residual_sums(h, d, L, alpha);
        SSE = SSE + s2 - 2*mu*s1 + L*mu^2;
        h = (d.^L) .* h;
    end
    
    % Occupied bin
    y = ycount / Delta;
    yhat = mu + alpha.' * h;
    SSE = SSE + (y - yhat)^2;
    
    h = d .* h + ycount;
    curr = tEvt + 1;
end

% Trailing zero run
L = T - curr + 1;
if L > 0
    [s1, s2] = zero_run_residual_sums(h, d, L, alpha);
    SSE = SSE + s2 - 2*mu*s1 + L*mu^2;
end

loss = SSE / T;

if ~isfinite(loss)
    loss = NaN;
end