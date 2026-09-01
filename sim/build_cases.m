function cases = build_cases(caseID,mu_value,targetBR,doIntegerA,alphaMin,betaMin)

%  Build capped and non-capped beta parameter cases while enforcing
%   min(A) >= alphaMin
%   min(B) >= betaMin.
%
% The chosen B vectors are sufficiently large so that the homogeneous and
% heterogeneous cases satisfy the lower bounds on alpha and beta.

cases = {};

switch caseID

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Non-capped beta designs
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case {'case1_1','case2_1'}

        B2 = [
            180
            16000
        ];

        B3 = [
            70
            750
            20000
        ];

        B4 = [
            250
            500
            2000
            20000
        ];

        % In K = 5 non-capped design: wider beta separation produces 
        % clearly separated alpha levels.
        B5 = [
            100
            500
            2000
            10000
            30000
        ];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Capped beta designs
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case {'case1_2','case2_2'}

        B2 = [
            200
            5000
        ];

        B3 = [
            70
            750
            5000
        ];

        B4 = [
            250
            500
            2000
            5000
        ];

        % In K = 5 capped design: beta_max is capped at 5000, while the
        % beta values are more widely separated.
        B5 = [
            100
            300
            1000
            3000
            5000
        ];

    otherwise
        error('Unknown caseID: %s',caseID);
end

% Ensure beta lower bound
if any(B2 < betaMin) || any(B3 < betaMin) || ...
        any(B4 < betaMin) || any(B5 < betaMin)

    error('At least one beta value is smaller than betaMin.');
end

scale_rho = targetBR/0.80;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K = 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rho_homo = scale_rho*[0.40;0.40];
rho_het  = scale_rho*[0.25;0.55];

cases = add_case( ...
    cases,mu_value,B2,rho_homo,targetBR, ...
    'homogeneous',doIntegerA,alphaMin,betaMin);

cases = add_case( ...
    cases,mu_value,B2,rho_het,targetBR, ...
    'heterogeneous',doIntegerA,alphaMin,betaMin);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K = 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rho_homo = scale_rho*[0.2667;0.2667;0.2666];
rho_het  = scale_rho*[0.15;0.25;0.40];

cases = add_case( ...
    cases,mu_value,B3,rho_homo,targetBR, ...
    'homogeneous',doIntegerA,alphaMin,betaMin);

cases = add_case( ...
    cases,mu_value,B3,rho_het,targetBR, ...
    'heterogeneous',doIntegerA,alphaMin,betaMin);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K = 4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rho_homo = scale_rho*[0.20;0.20;0.20;0.20];
rho_het  = scale_rho*[0.08;0.14;0.23;0.35];

cases = add_case( ...
    cases,mu_value,B4,rho_homo,targetBR, ...
    'homogeneous',doIntegerA,alphaMin,betaMin);

cases = add_case( ...
    cases,mu_value,B4,rho_het,targetBR, ...
    'heterogeneous',doIntegerA,alphaMin,betaMin);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K = 5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rho_homo = scale_rho*[0.16;0.16;0.16;0.16;0.16];
rho_het  = scale_rho*[0.03;0.07;0.12;0.23;0.35];

cases = add_case( ...
    cases,mu_value,B5,rho_homo,targetBR, ...
    'homogeneous',doIntegerA,alphaMin,betaMin);

cases = add_case( ...
    cases,mu_value,B5,rho_het,targetBR, ...
    'heterogeneous',doIntegerA,alphaMin,betaMin);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Final validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for cc = 1:numel(cases)

    A = cases{cc}.A(:);
    B = cases{cc}.B(:);

    actualBR = sum(A./B);

    if min(A) < alphaMin
        error( ...
            'Case %d violates alpha lower bound: min(A) = %.12g.', ...
            cc,min(A));
    end

    if min(B) < betaMin
        error( ...
            'Case %d violates beta lower bound: min(B) = %.12g.', ...
            cc,min(B));
    end

    if actualBR < targetBR - 1e-12 || actualBR >= 1
        error( ...
            'Case %d has invalid BR %.12g.',cc,actualBR);
    end
end