function [sA, sA2] = zero_run_residual_sums(h0,d,L,alpha)

% For zero-count bins, y=0 and yhat_l = mu + alpha' h_l.
% This function returns:
%   sA  = sum_l alpha' h_l
%   sA2 = sum_l (alpha' h_l)^2
% where h_l = d^(l-1) .* h0, l=1,...,L.

g1 = geom_sum(d, L);
sA = alpha.' * (h0 .* g1);

dd = d * d.';
G2 = geom_sum(dd, L);

Q = (h0 * h0.') .* G2;
sA2 = alpha.' * Q * alpha;

if ~isfinite(sA)
    sA = NaN;
end
if ~isfinite(sA2)
    sA2 = NaN;
end