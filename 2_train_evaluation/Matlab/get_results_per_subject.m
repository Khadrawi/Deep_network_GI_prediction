function results_per_animal = get_results_per_subject(results_struct, labels, subject_labels, participants)

results_per_animal = table('Size',[0 , 4],'VariableTypes',{'double','double','double','double'}, 'VariableNames',participants);
% 9 for number of classifiers(includingCNN), 3 for acc,sens,spec
tmp_acc = zeros(1, length(participants));
tmp_sens = zeros(1, length(participants));
tmp_spec = zeros(1, length(participants));
for i = 1:8 % number of methods length(model_names)
     tmp_score = results_struct(i).scores;
     
     for j = 1:length(participants)
        idx = subject_labels == participants(j);
        res = tmp_score(idx) >= 0.5;
        conf = confusionmat(logical(labels(idx)), res);
        tmp_acc(j) = (conf(1,1)+conf(2,2))*100/sum(conf, 'All');
        sums = sum(conf, 2);
        tmp_sens(j) = conf(2,2)*100/sums(2);
        tmp_spec(j) = conf(1,1)*100/sums(1);
     end
     tmp_table = array2table([tmp_acc; tmp_sens; tmp_spec], 'VariableNames',participants);
     results_per_animal = [results_per_animal; tmp_table];
end

end