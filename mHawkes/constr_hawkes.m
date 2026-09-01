function [c,ceq] = constr_hawkes(theta,B)

%  OUTPUTS:
%    C    : Vector of nonlinear inequality constraints.  Based on the roots
%           of a polynomial in beta
%    CEQ  : Empty matrix

mu = theta(1); A = theta(2:end); B = B(:);
c = [ 
    1e-8-mu; 
    mu-100;
    1e-8-A; 
    1e-8+A-max(B); 
    sum(A./B)-(1-1e-6)]; 
ceq=[]; 