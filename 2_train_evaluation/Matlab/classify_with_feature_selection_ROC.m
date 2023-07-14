function [fpr, tpr, auc, scores, fs, performance] = classify_with_feature_selection_ROC(x,y,numOfFolds,C,classifier_type, direction)
% input: x ,y data out ..., fs selected features needs table creation, C
% specificity penalty for eature selection
% output: accuracy, sensitivity, specificity, fs, fs is logical index vector for
% chosen features from sequential feature selection

classifier = @(train_data,train_labels,test_data,test_labels)multi_classifier(train_data,train_labels,test_data,test_labels,C,classifier_type);

% standardize   
%  x=zscore(x);

cv = cvpartition(y,'k',numOfFolds, 'Stratify',false);%
opts = struct('UseParallel',true);
% opts = statset('display','iter');%iter

% Sequential feature selection
[fs,~] = sequentialfs(classifier,x,y,'cv',cv,'mcreps',100,'options',opts,'direction', direction);

cv = cvpartition(y,'k',numOfFolds, 'Stratify',false);%
scores_cell = {};
accuracy = zeros(1,3); sensitivity = zeros(1,3); specificity = zeros(1,3);

parfor j=1:numOfFolds
    train_data = x(cv.training(j),fs);
    [train_data, mu, sigma] = zscore(train_data);
    test_data = x(cv.test(j),fs);
    test_data = (test_data-mu)./sigma;
    train_labels = y(cv.training(j));
    test_labels = y(cv.test(j));
    opts = struct('Optimizer','bayesopt','ShowPlots',false,...
    'AcquisitionFunctionName','expected-improvement-plus','Verbose',0);
    switch classifier_type
        case "SVM - Linear"
            mdl = fitcsvm(train_data,train_labels,'KernelFunction','linear','KernelScale','auto','Standardize',false,'Cost',[0 C(1); C(2) 0], 'solver', 'SMO');%, 'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
            mdl = fitPosterior(mdl);
            [~, res] = predict(mdl, test_data);
        case "SVM - Polynomial 2"
            mdl = fitcsvm(train_data,train_labels,'KernelFunction','polynomial','PolynomialOrder', 2,'KernelScale','auto','Standardize',false,'Cost',[0 C(1); C(2) 0], 'solver', 'SMO');
            mdl = fitPosterior(mdl);
            [~, res] = predict(mdl, test_data);
        case "SVM - Polynomial 3"
            mdl = fitcsvm(train_data,train_labels,'KernelFunction','polynomial','PolynomialOrder', 3,'KernelScale','auto','Standardize',false,'Cost',[0 C(1); C(2) 0], 'solver', 'SMO');
            mdl = fitPosterior(mdl);
            [~, res] = predict(mdl, test_data);
        case "SVM - RBF"
            mdl = fitcsvm(train_data,train_labels,'KernelFunction','rbf','KernelScale','auto','Standardize',false,'Cost',[0 C(1); C(2) 0], 'solver', 'SMO');
            mdl = fitPosterior(mdl);
            [~, res] = predict(mdl, test_data);
        case "Regularized LDA"
            mdl = fitcdiscr(train_data,train_labels,'Cost', [0 C(1); C(2) 0], 'OptimizeHyperparameters',{'Gamma','Delta'},'DiscrimType', 'Linear','HyperparameterOptimizationOptions',opts);
            [~, res, ~] = predict(mdl, test_data);
        case "QDA"
            mdl = fitcdiscr(train_data,train_labels,'Cost', [0 C(1); C(2) 0],'DiscrimType', 'Quadratic');   
            [~, res, ~] = predict(mdl, test_data);
        case "Logistic Regression"    
            mdl = fitclinear(train_data,train_labels, 'learner', 'logistic','OptimizeHyperparameters','lambda','HyperparameterOptimizationOptions',opts, 'Cost', [0 C(1); C(2) 0]);    
            [~, res] = predict(mdl, test_data);
        case "Random Forest"
            mdl = fitcensemble(train_data,train_labels, 'Cost', [0 C(1); C(2) 0]);
            [~, res] = predict(mdl, test_data);
    end
    scores_cell{j} = res(:,2);
    predicted_labels = predict(mdl, test_data);
    conf = confusionmat(test_labels, predicted_labels);
    accuracy(j) = (conf(1,1)+conf(2,2))*100/sum(conf, 'All');
    sums = sum(conf, 2);
    sensitivity(j) = conf(2,2)*100/sums(2);
    specificity(j) = conf(1,1)*100/sums(1);
end
scores = zeros(length(y),1);
% Adjustment for parfor slicing limitation
for i = 1:3
    scores(cv.test(i)) = scores_cell{i};
end
[fpr,tpr,~,auc] = perfcurve(y, scores, 1);
performance = struct();
performance.Accuracy_Avg = mean(accuracy);
performance.Accuracy_SD = std(accuracy);
performance.Sensitivity_Avg = mean(sensitivity);
performance.Sensitivity_SD = std(sensitivity);
performance.Specificity_Avg = mean(specificity);
performance.Specificity_SD = std(specificity);
end