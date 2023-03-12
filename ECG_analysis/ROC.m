function ROC = ROC(data, dataClass, x)
predictClass = zeros(size(data));
ROC = zeros(2, length(x));
for j = 1:length(x)
    for i=1:length(data)
        if data(i) > x(j)
            predictClass(i) = 1;
        else
            predictClass(i) = 2;
        end
    end
    [~, sens, spec, ~, ~, ~, ~] = accuracy(predictClass, dataClass);
    ROC(:,j)=[1-spec; sens];
end
end