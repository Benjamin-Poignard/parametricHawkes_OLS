function [eventsTimes, lambdaPath, tGrid, info] = simulate_mhawkes(mu,alpha,beta,Tmax,opts)

% Univariate Hawkes with multiple exponential kernels.
% Model: lambda(t) = mu + sum_{k=1}^K alpha_k * sum_{t_j < t} exp(-beta_k * (t - t_j))
%
% INPUTS:
%   mu     : scalar >= 0
%   alpha  : K x 1 or 1 x K, alpha_k >= 0
%   beta   : K x 1 or 1 x K, beta_k > 0
%   Tmax   : horizon
%   opts.seed      : RNG seed (default NaN)
%   opts.maxEvents : safety cap (default 2e5)
%   opts.dt        : grid step for lambdaPath (default 0.01)
%
% OUTPUTS:
%   eventsTimes : N x 1 event times
%   lambdaPath  : Tg x 1 intensity on grid tGrid
%   tGrid       : Tg x 1 regular grid
%   info        : struct with diagnostics

arguments
    mu (1,1) double
    alpha (:,1) double
    beta  (:,1) double
    Tmax (1,1) double {mustBePositive}
    opts.seed (1,1) double = NaN
    opts.maxEvents (1,1) double {mustBePositive} = 2e5
    opts.dt (1,1) double {mustBePositive} = 0.01
end

K = numel(alpha);
assert(numel(beta) == K, 'alpha and beta must have same length.');
assert(mu >= 0, 'mu must be >= 0.');
assert(all(alpha >= 0), 'alpha must be >= 0.');
assert(all(beta > 0), 'beta must be > 0.');

if ~isnan(opts.seed)
    rng(opts.seed);
end

% --- Grid for intensity trajectory ---
dt = opts.dt;
tGrid = (0:dt:Tmax)';
Tg = numel(tGrid);
lambdaPath = zeros(Tg,1);

% State:
% g_k(t) = sum_{t_j < t} exp(-beta_k * (t - t_j))
g = zeros(K,1);
t = 0;

eventsTimes = zeros(0,1);

lambda_from_g = @(gcur) mu + sum(alpha .* gcur);

% initial intensity
lambda = lambda_from_g(g);
lambdaPath(1) = lambda;

gridIdx = 2;
nEvents = 0;

while t < Tmax && nEvents < opts.maxEvents
    
    LambdaBar = lambda;
    if LambdaBar <= 0
        break;
    end
    
    % Candidate time
    w = -log(rand) / LambdaBar;
    tCand = t + w;
    if tCand > Tmax
        break;
    end
    
    % Fill grid points between current t and candidate time
    while gridIdx <= Tg && tGrid(gridIdx) < tCand
        wg = tGrid(gridIdx) - t;
        gAt = g .* exp(-beta * wg);
        lambdaPath(gridIdx) = lambda_from_g(gAt);
        gridIdx = gridIdx + 1;
    end
    
    % Decay state to candidate time
    gCand = g .* exp(-beta * w);
    lambdaCand = lambda_from_g(gCand);
    
    % Thinning
    if rand <= lambdaCand / LambdaBar
        % Accept
        nEvents = nEvents + 1;
        t = tCand;
        eventsTimes(end+1,1) = t; %#ok<AGROW>
        
        % Jump update: every kernel receives +1
        g = gCand + 1;
        lambda = lambda_from_g(g);
        
        while gridIdx <= Tg && abs(tGrid(gridIdx) - t) < 1e-12
            lambdaPath(gridIdx) = lambda;
            gridIdx = gridIdx + 1;
        end
    else
        % Reject
        t = tCand;
        g = gCand;
        lambda = lambdaCand;
        
        while gridIdx <= Tg && abs(tGrid(gridIdx) - t) < 1e-12
            lambdaPath(gridIdx) = lambda;
            gridIdx = gridIdx + 1;
        end
    end
end

% Fill remaining grid points until Tmax
while gridIdx <= Tg
    wg = tGrid(gridIdx) - t;
    gAt = g .* exp(-beta * wg);
    lambdaPath(gridIdx) = lambda_from_g(gAt);
    gridIdx = gridIdx + 1;
end

info = struct();
info.K = K;
info.nEvents = numel(eventsTimes);
info.meanRate = numel(eventsTimes) / Tmax;
info.hitMaxEvents = (nEvents >= opts.maxEvents);
info.dt = dt;
info.Tg = Tg;