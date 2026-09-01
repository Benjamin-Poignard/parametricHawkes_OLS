function [Vsum,hEnd,gHist] = process_zero_run(Vsum,h0,gHist,L,beta,Delta,d,theta,mT,weights,mono,useParallelForLags)

% Processes a consecutive run of L zero-count bins when constructing
% the HAC covariance estimator for the discretized Hawkes OLS model.
%
% INPUTS:
%   Vsum : current p x p accumulated HAC covariance matrix.
%   h0   : K x 1 Hawkes state vector at the beginning of the zero run.
%   gHist: matrix storing the most recent moment vectors g_t needed
%          for HAC lag calculations. Each row contains one past
%          p-dimensional moment vector.
%   L    : number of consecutive zero-count bins to process.
%   beta : K x 1 vector of exponential decay parameters.
%   Delta: positive bin width.
%   d    : K x 1 vector of one-bin decay factors d_k = exp(-beta_k*Delta).
%   theta: p x 1 OLS parameter vector theta = [mu,alpha_1, ..., alpha_K], where p = K+1.
%   mT   : maximum HAC lag.
%   weights: vector of HAC lag weights. weights(tau+1) is the weight
%            associated with lag tau, for tau=0,...,mT.
%   mono : structure containing precomputed quantities used by
%          zero_run_hac_block to evaluate long zero runs analytically.
%   useParallelForLags: logical flag controlling whether the lag computations in
%          zero_run_hac_block are evaluated using parfor.
%
% OUTPUTS:
%   Vsum : updated p x p accumulated HAC covariance matrix after
%          accounting for the complete zero run.
%   hEnd : K x 1 Hawkes state vector at the end of the zero run.
%   gHist: updated history of the most recent moment vectors required
%          for subsequent HAC lag calculations.

if L <= 0
    hEnd = h0;
    return;
end

p = numel(beta) + 1; Lexplicit = min(L,mT); h = h0;

for ell = 1:Lexplicit
    [Vsum,h,gHist] = process_one_bin( ...
        Vsum,h,gHist,0,Delta,d,theta,mT,weights);
end

if Lexplicit < L
    lStart = Lexplicit;
    lEnd = L - 1;
    
    Vsum = Vsum + zero_run_hac_block( ...
        h0,lStart,lEnd,mT,weights,mono,p,useParallelForLags);
end

hEnd = h0 .* exp(-beta * Delta * L);

if mT > 0
    lastCount = min(mT,L);
    firstLocal = L - lastCount;
    
    newHist = zeros(lastCount,p);
    
    decayMat = exp(-(firstLocal:(L-1)).' * (beta(:).' * Delta));
    hMat = decayMat .* h0(:).';
    
    for rr = 1:lastCount
        x = [1; hMat(rr,:).'];
        u = -theta.' * x;
        newHist(rr,:) = (x * u).';
    end
    
    gHist = newHist;
end