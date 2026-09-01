function s = geom_sum(r,L)

% Computes the finite geometric sum s = sum_{ell=0}^{L-1} r^ell
% elementwise for all entries of r.
%
% INPUTS:
%   r : scalar, vector, or matrix containing the common ratios of
%       the geometric sums. The computation is elementwise.
%   L : number of terms in the geometric sum.
%
% OUTPUT:
%   s : array having the same size as r, with s(i) = sum_{ell=0}^{L-1} r(i)^ell.
%       For r sufficiently close to 1, the function sets s=L
%       to avoid numerical instability.

tol = 1e-12; s = zeros(size(r));
mask = abs(1 - r) > tol;
s(mask) = (1 - r(mask).^L) ./ (1 - r(mask));
s(~mask) = L;