function x = clean_event_times(x)

x = x(:);
x = x(isfinite(x));
x = sort(x);

if isempty(x)
    return;
end

if any(diff(x) == 0)
    dupIdx = find(diff(x) == 0);
    for jj = 1:numel(dupIdx)
        x(dupIdx(jj)+1) = x(dupIdx(jj)+1) + 1e-9 * jj;
    end
    x = sort(x);
end