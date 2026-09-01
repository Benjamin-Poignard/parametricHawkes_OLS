function Vtau = zero_run_one_tau(baseVal,tau0,lStart,weights,mono,p)

if tau0 > lStart
    Vtau = zeros(p,p);
    return;
end

vals = baseVal .* exp(mono.Delta * tau0 * mono.lagBetaSum);

Sflat = accumarray(mono.outIdx,vals,[p*p,1],@sum,0);
S = reshape(Sflat,p,p);

if tau0 == 0
    Vtau = weights(1) * S;
else
    Vtau = weights(tau0+1) * (S + S.');
end