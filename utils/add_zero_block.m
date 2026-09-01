function XtX = add_zero_block(XtX,h,phi,g)

K = numel(h);
XtX(1,1) = XtX(1,1) + g;
Sh = h .* geom_sum(phi,g);
XtX(1,2:end) = XtX(1,2:end) + Sh.';
XtX(2:end,1) = XtX(2:end,1) + Sh;

for i = 1:K
    for j = 1:K
        r = phi(i)*phi(j);
        XtX(i+1,j+1) = XtX(i+1,j+1) + h(i)*h(j)*geom_sum(r,g);
    end
end