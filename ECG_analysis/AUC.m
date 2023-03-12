function [acc, sens, spec, TP, TN, FP, FN] = AUC(V1, V2, x, a)
delta = x(2)-x(1);
S1 = sum(V1.*delta);
S2 = sum(V2.*delta);

TN = sum(V1(x>=a).*delta)/S1;
TP = sum(V2(x<a).*delta)/S2;
FP = 1-TN;  
FN = 1-TP;

acc = (TP+TN)/2;
sens = TP/(FN+TP);
spec = TN/(FP+TN);
end