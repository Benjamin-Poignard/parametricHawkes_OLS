function markerFlag = marker_flags_for_smaller_n(mleBase,n_store,threshold)

N = numel(mleBase); markerFlag = false(N,1);

for a = 1:N
    for b = a+1:N
        
        if ~isfinite(mleBase(a)) || ~isfinite(mleBase(b))
            continue;
        end
        
        if abs(mleBase(a) - mleBase(b)) < threshold
            
            if n_store(a) < n_store(b)
                markerFlag(a) = true;
            elseif n_store(b) < n_store(a)
                markerFlag(b) = true;
            end
        end
    end
end