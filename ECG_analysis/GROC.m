function ROC = GROC(V1, V2, x)
ROC = zeros(2,length(x));
for i=1:length(x)
    [~, sens, spec, ~, ~, ~, ~] = AUC(V1, V2, x, x(i));
    ROC(:,i)=[1-spec; sens];
end
end