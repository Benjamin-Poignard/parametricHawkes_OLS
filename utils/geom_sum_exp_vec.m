function s = geom_sum_exp_vec(a,L)

% Computes the finite geometric sum s = sum_{ell=0}^{L-1} exp(-a*ell)
% elementwise for all entries of a.
%
% INPUTS:
%   a : scalar, vector, or matrix of exponential decay-rate arguments.
%       The computation is performed elementwise.
%   L : nonnegative integer giving the number of terms in the
%       geometric sum.
%
% OUTPUT:
%   s : array having the same size as a, with s(i) = sum_{ell=0}^{L-1} exp(-a(i)*ell).
%       If L <= 0, s is identically zero.
%       For a close to zero, the function sets s=L to avoid
%       numerical instability.

s = zeros(size(a));
if L <= 0
    return;
end

small = abs(a) < 1e-14;
s(small) = L;

idx = ~small;
s(idx) = -expm1(-a(idx) * L) ./ (-expm1(-a(idx)));