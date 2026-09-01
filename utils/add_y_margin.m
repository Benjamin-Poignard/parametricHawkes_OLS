function add_y_margin(ax,marginRate)

h = findobj(ax,'Type','line'); ymin = inf; ymax = -inf;

for ii = 1:numel(h)
    y = h(ii).YData;
    y = y(isfinite(y));

    if ~isempty(y)
        ymin = min(ymin, min(y));
        ymax = max(ymax, max(y));
    end
end

yrange = ymax - ymin;

if yrange == 0
    yrange = max(abs(ymax),1);
end

margin = marginRate * yrange;
ylim(ax, [ymin - margin, ymax + margin]);