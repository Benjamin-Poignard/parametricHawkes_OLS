function mono = make_zero_monomial_table(beta,Delta,theta)

beta = beta(:);theta = theta(:); K = numel(beta);p = K + 1;
M = p^4;
outIdx = zeros(M,1);
thetaCoef = zeros(M,1);
betaSum = zeros(M,1);
lagBetaSum = zeros(M,1);
hPowCount = zeros(M,K);

mm = 0;

for a = 1:p
    for b = 1:p
        for rr = 1:p
            for q = 1:p
                
                mm = mm + 1;
                
                outIdx(mm) = sub2ind([p,p],a,b);
                thetaCoef(mm) = theta(rr) * theta(q);
                
                idxCurrent = [a,rr] - 1;
                idxLagged  = [b,q] - 1;
                
                idxCurrent = idxCurrent(idxCurrent > 0);
                idxLagged  = idxLagged(idxLagged > 0);
                
                if ~isempty(idxCurrent)
                    betaSum(mm) = betaSum(mm) + sum(beta(idxCurrent));
                    
                    for kk = idxCurrent(:).'
                        hPowCount(mm,kk) = hPowCount(mm,kk) + 1;
                    end
                end
                
                if ~isempty(idxLagged)
                    betaSum(mm) = betaSum(mm) + sum(beta(idxLagged));
                    lagBetaSum(mm) = sum(beta(idxLagged));
                    
                    for kk = idxLagged(:).'
                        hPowCount(mm,kk) = hPowCount(mm,kk) + 1;
                    end
                end
            end
        end
    end
end

mono = struct();
mono.outIdx = outIdx;
mono.thetaCoef = thetaCoef;
mono.betaSum = betaSum;
mono.lagBetaSum = lagBetaSum;
mono.hPowCount = hPowCount;
mono.aRate = Delta * betaSum;
mono.Delta = Delta;