function Ainvhalf = sym_inv_sqrt(A,eigFloor)

A = (A + A.') / 2;
[Q,Lam] = eig(A);
lam = real(diag(Lam));

scale = max(max(abs(lam)),1);
floorVal = eigFloor * scale;
lamFloor = max(lam,floorVal);

Ainvhalf = Q * diag(1 ./ sqrt(lamFloor)) * Q.';
Ainvhalf = (Ainvhalf + Ainvhalf.') / 2;