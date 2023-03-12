function [acc, sens, spec, TP, TN, FP, FN] = accuracy(predictClass, dataClass)
TP = length(find(predictClass ~= 1 & dataClass ~= 1));
TN = length(find(predictClass == 1 & dataClass == 1));
FP = length(find(predictClass ~= 1 & dataClass == 1));  
FN = length(find(predictClass == 1 & dataClass ~= 1));

acc = (TP + TN) / height(dataClass);
sens = TP / (FN + TP);
spec = TN / (FP + TN);
end