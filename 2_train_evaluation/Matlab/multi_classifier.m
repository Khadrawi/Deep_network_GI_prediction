function mis_score = multi_classifier(train_data,train_labels,test_data,test_labels,C, classifier_type)
% C = [C1, C2], C1 -> ground(==0) ~= predicted penalty (specificity penalty),
% C2 - > gound(==1) ~= predicted penalty (sensitivity penalty)

[train_data, mu, sigma] = zscore(train_data);
test_data = (test_data-mu)./sigma;
switch classifier_type
    case "SVM - Linear"
        mdl = fitcsvm(train_data,train_labels,'KernelFunction','linear','Standardize',false,'Cost',[0 C(1); C(2) 0], 'solver', 'SMO');%, 'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
        res = predict(mdl, test_data);
    case "SVM - Polynomial 2"
        mdl = fitcsvm(train_data,train_labels,'KernelFunction','polynomial','PolynomialOrder', 2,'Standardize',false,'Cost',[0 C(1); C(2) 0], 'solver', 'SMO');
        res = predict(mdl, test_data);
    case "SVM - Polynomial 3"
        mdl = fitcsvm(train_data,train_labels,'KernelFunction','polynomial','PolynomialOrder', 3,'Standardize',false,'Cost',[0 C(1); C(2) 0], 'solver', 'SMO');
        res = predict(mdl, test_data);
    case "SVM - RBF"
        mdl = fitcsvm(train_data,train_labels,'KernelFunction','rbf','KernelScale','auto','Standardize',false,'Cost',[0 C(1); C(2) 0], 'solver', 'SMO');
        res = predict(mdl, test_data);
    case "Regularized LDA"
        mdl = fitcdiscr(train_data,train_labels,'Cost', [0 C(1); C(2) 0],'DiscrimType', 'Linear');
        res = predict(mdl, test_data);
    case "QDA"
        mdl = fitcdiscr(train_data,train_labels,'Cost', [0 C(1); C(2) 0],'DiscrimType', 'Quadratic');   
        res = predict(mdl, test_data);
    case "Logistic Regression"    
        mdl = fitclinear(train_data,train_labels, 'learner', 'logistic', 'Cost', [0 C(1); C(2) 0]);    
        res = predict(mdl, test_data);
    case "Random Forest"
        mdl = fitcensemble(train_data,train_labels, 'Cost', [0 C(1); C(2) 0]);
        res = predict(mdl, test_data);
end

% score = 0;
% for i = 1:length(test_labels)
%     if res(i) ~= test_labels(i)
%         if test_labels(i) == 0%improve specificity
%             score = score + C;
%         else
%             score = score + 1;
%         end
%     end

% Get misclassification score
miss = logical(test_labels(res ~= test_labels));
% Specificity penalty
spec = C(1)* sum(~miss);
% Sensitivity penalty
sens = C(2)*sum(miss);
mis_score = sens + spec;

end