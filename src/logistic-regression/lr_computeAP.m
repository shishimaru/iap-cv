function [AP] = lr_computeAP(classes, thetas, X, y)

%% Compute accuracy on our validation set
accuracies = zeros(size(classes,2),1);
for i=1:size(classes,2)
    prob = X * thetas(:,i);
    AP = computeAP(prob, y(:,i), 1)*100;
    accuracies(i, 1) = AP;
    fprintf('Val Accuracy (%13s) : %0.3f%%\n', classes{i}, AP);
end;
AP = mean(accuracies);
fprintf('Val Accuracy Average : %0.3f%%\n', AP);

end