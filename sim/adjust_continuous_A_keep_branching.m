function A = adjust_continuous_A_keep_branching(A,B,targetBR,alphaMin)

% Continuous-valued counterpart used only when doIntegerA is false.

A = A(:); B = B(:); A = max(A,alphaMin); eta = sum(A./B);

if eta >= 1

    excess = eta-(1-1e-8);

    reducible = A-alphaMin;
    availableBR = sum(reducible./B);

    if availableBR <= excess
        error(['Cannot restore stationarity without violating ' ...
            'the alpha lower bound.']);
    end

    % Reduce proportionally to available reducible mass
    reductionShare = excess/availableBR;
    A = A-reductionShare*reducible;

    eta = sum(A./B);
end

if eta < targetBR

    deficit = targetBR-eta;

    % Add mass to largest beta, creating the smallest change per unit alpha
    [~,idx] = max(B);
    A(idx) = A(idx)+deficit*B(idx);

    eta = sum(A./B);
end

if eta >= 1
    error('Continuous adjustment failed to preserve stationarity.');
end