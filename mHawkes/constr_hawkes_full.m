function [c,ceq] = constr_hawkes_full(theta,K)

%  OUTPUTS:
%    C    : Vector of nonlinear inequality constraints.  Based on the roots
%           of a polynomial in beta.
%    CEQ  : Empty matrix.

mu = theta(1); theta = theta(2:end);
A = theta(1:K); B = theta(K+1:end);
c = [
    1e-8-mu;
    mu-100;
    1e-8-A;
    1e-8-B;
    1e-8+A-max(B);
    sum(A./B)-(1-1e-6)];
ceq=[];