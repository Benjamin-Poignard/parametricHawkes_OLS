function cases = add_case(cases,mu,B,rho,targetBR,typeStr,doIntegerA,alphaMin,betaMin)

% Code for simulation experiments on l2 error
B = B(:); rho = rho(:);

if any(B < betaMin)
    error('The beta lower-bound restriction is violated.');
end

Araw = rho.*B;

if doIntegerA
    
    A = round_integer_A_keep_branching( ...
        Araw,B,targetBR,alphaMin);
    
else
    
    A = max(Araw,alphaMin);
    A = adjust_continuous_A_keep_branching( ...
        A,B,targetBR,alphaMin);
end

actualBR = sum(A./B);

if any(A < alphaMin)
    error('The alpha lower-bound restriction is violated.');
end

if actualBR < targetBR - 1e-12 || actualBR >= 1
    error(['Adjusted branching ratio %.12g is outside ' ...
        '[%.12g,1).'],actualBR,targetBR);
end

cases{end+1} = struct( ...
    'mu',mu, ...
    'A',A(:), ...
    'B',B(:), ...
    'type',typeStr, ...
    'targetBR',targetBR, ...
    'actualBR',actualBR, ...
    'minAlpha',min(A), ...
    'minBeta',min(B));
end


function A = round_integer_A_keep_branching( ...
    Araw,B,targetBR,alphaMin)
% Round all alpha values, enforce A >= alphaMin, and then adjust the
% largest-beta component to obtain targetBR <= sum(A./B) < 1.

B = B(:);
Araw = Araw(:);

A = round(Araw);
A = max(A,alphaMin);

eta = sum(A./B);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ensure stationarity if rounding/flooring gives BR >= 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while eta >= 1
    
    reducibleIndex = find(A > alphaMin);
    
    if isempty(reducibleIndex)
        error(['Cannot restore stationarity without violating ' ...
            'the alpha lower bound.']);
    end
    
    % Reduce the component with the largest beta first
    [~,jj] = max(B(reducibleIndex));
    idx = reducibleIndex(jj);
    
    A(idx) = A(idx)-1;
    eta = sum(A./B);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Raise Branching ratio (BR) to target if necessary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while eta < targetBR
    
    feasibleIndex = find(eta + 1./B < 1);
    
    if isempty(feasibleIndex)
        error(['No feasible integer increment can reach target BR ' ...
            'while preserving stationarity.']);
    end
    
    % Increase the component with the largest beta first
    [~,jj] = max(B(feasibleIndex));
    idx = feasibleIndex(jj);
    
    A(idx) = A(idx)+1;
    eta = sum(A./B);
end