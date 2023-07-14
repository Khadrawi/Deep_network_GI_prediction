function score = classf(train_data,train_labels,test_data,test_labels,C)
% C = [C1, C2], C1 -> ground(==0) ~= predicted penalty (specificity penalty),
% C2 - > gound(==1) ~= predicted penalty (sensitivity penalty)
opts = struct('Optimizer','bayesopt','ShowPlots',false,...
    'AcquisitionFunctionName','expected-improvement-plus','Verbose',0);
[train_data, mu, sigma] = zscore(train_data);
% classifier = fitcensemble(X_train, train_labels, 'Cost',[0 C(1); C(2) 0]);
% classifier = fitclinear(X_train, train_labels, 'learner', 'logistic', 'Cost',[0 C(1); C(2) 0]);
% classifier = fitcdiscr(X_train, train_labels, 'DiscrimType', 'Linear','Cost',[0 C(1); C(2) 0]);
% classifier = fitcdiscr(X_train, train_labels, 'DiscrimType', 'Quadratic','Cost',[0 C(1); C(2) 0]);
% classifier = fitcsvm(X_train,train_labels,'KernelFunction','polynomial','PolynomialOrder', 3, 'Standardize',false);%,'PolynomialOrder',5);%,
% classifier = fitcsvm(train_data,train_labels,'KernelFunction','polynomial','PolynomialOrder',2, 'Standardize',true);
classifier = fitcsvm(train_data,train_labels,'KernelFunction','rbf', 'Standardize',false);
% classifier = fitcknn(train_data,train_labels,'NumNeighbors',3);
res = predict(classifier , (test_data-mu)./sigma);

% score = 0;
% for i = 1:length(test_labels)
%     if res(i) ~= test_labels(i)
%         if test_labels(i) == 0%improve specificity
%             score = score + C;
%         else
%             score = score + 1;
%         end
%     end
miss = logical(test_labels(res ~= test_labels));
% Specificity penalty
spec = C(1)* sum(~miss);
% Sensitivity penalty
sens = C(2)*sum(miss);
score = sens + spec;

end