function W = compute_wald_from_base(outBase,R,r)

% Computes a Wald test statistic for the linear hypothesis H0: R * theta = r
% using a previously computed parameter estimate and covariance matrix.
%
% INPUTS:
%   outBase : structure containing the quantities required for the
%             Wald statistic:
%                 outBase.theta_hat: p x 1 estimated parameter vector.
%                 outBase.Dhat: p x p estimated covariance matrix associated with theta_hat.
%                 outBase.Tbins: sample size measured as the number of bins.
%                 outBase.TDelta: alternative sample-size/time scaling.
%                 outBase.scaleUsed: string specifying which scaling is used for the reported Wald statistic:
%                         'tbins', 't', 'eq22' -> use Tbins;
%                         'tdelta', 't_delta', 'td' -> use TDelta.
%   R       : q x p restriction matrix defining the linear hypothesis H0: R*theta = r.
%   r       : q x 1 vector containing the null-hypothesis values.
%
% OUTPUT:
%   W       : structure containing the Wald-test results:
%                 W.S: Wald statistic using the scaling selected by outBase.scaleUsed.
%                 W.S_Tbins: Wald statistic scaled by Tbins.
%                 W.S_TDelta: Wald statistic scaled by TDelta.
%                 W.q: number of tested restrictions, i.e. size(R,1).
%                 W.p: chi-square p-value 1 - chi2cdf(W.S,q).
%                 W.Rtheta_minus_r: q x 1 vector R*theta_hat - r.
%                 W.RDhatR: q x q estimated covariance matrix of the restricted parameter combination: R*Dhat*R'.

R = double(R); r = r(:);
theta_hat = outBase.theta_hat(:);
Dhat = outBase.Dhat;

q = size(R,1); d = R * theta_hat - r;
RDhatR = R * Dhat * R.';
RDhatR = (RDhatR + RDhatR.') / 2;

if q == 1
    RDhatRinv = 1 / RDhatR;
else
    RDhatRinv = RDhatR \ eye(q);
end

quad = real(d.' * RDhatRinv * d);

S_Tbins  = outBase.Tbins  * quad;
S_TDelta = outBase.TDelta * quad;

switch lower(outBase.scaleUsed)
    case {'tbins','t','eq22'}
        S = S_Tbins;
    case {'tdelta','t_delta','td'}
        S = S_TDelta;
    otherwise
        error('Unknown scaleUsed.');
end

W = struct();
W.S = S;
W.S_Tbins = S_Tbins;
W.S_TDelta = S_TDelta;
W.q = q;
W.p = 1 - chi2cdf(S,q);
W.Rtheta_minus_r = d;
W.RDhatR = RDhatR;