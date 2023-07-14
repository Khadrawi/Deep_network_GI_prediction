%% Read preprocessed data
clear
% base_path = '..\..\data';
base_path = '..\..\preprocessed data';
path = dir(base_path);
path(1:2)=[];
s = struct('feeding', 'baseline');
segment_length = 12000;
num_segments = 60;
participants = ["103-21", "58-21", "60-21", "87-21"];
num_participants = length(participants);
num_channels = 4;
segmented_sig = cell(num_segments*num_participants*2,1);
f_s=200;
labels = zeros(num_segments*num_participants*2,1);
subject_labels = strings(num_segments*num_participants*2,1);
particip_label = repelem([1:num_participants], num_segments*2 );
dim1 = ones(1, num_segments)*segment_length; dim2 = ones(1,60)*6;

for i = 1:num_participants
    % READ 720000 points at fs = 200
    % read sub

    file = fullfile(base_path, participants(i));
    sub_path = dir(file);
    sub_path(1:2) = [];
    sub_files = {sub_path(:).name};
    
    % feeding file
    feeding_file_idx = contains(sub_files, 'feed') & contains(sub_files, 'csv');
    feeding_file_name = fullfile(file, sub_files{feeding_file_idx});
    feeding_table = readtable(feeding_file_name);
    data_f = feeding_table{1:720000,2:num_channels+1};
    segmented_sig(num_segments*(2*i-2)+1:num_segments*(2*i-1)) = mat2cell(data_f, dim1);
    labels(num_segments*(2*i-2)+1:num_segments*(2*i-1)) = 1; % Feeding label = 1
    
    % baseline file
    feeding_file_idx = contains(sub_files, 'base') & contains(sub_files, 'csv');
    baseline_file_name = fullfile(file, sub_files{feeding_file_idx});
    baseline_table = readtable(baseline_file_name);
    data_b = baseline_table{1:720000,2:num_channels+1};
    segmented_sig(num_segments*(2*i-1)+1:num_segments*(2*i)) = mat2cell(data_b, dim1);
    subject_labels(num_segments*(2*i-2)+1:num_segments*(2*i)) = participants(i);
end

varnames= string(feeding_table.Properties.VariableNames);
%% Get PSD
start_freq = 0.05;
end_freq = 0.7;
total_band_len = end_freq - start_freq; % band of frequencies to be analyzed Hz
num_freq_bands = 10; % numbers of cuts of the total band
one_band_length = total_band_len / num_freq_bands;
band_start_hz = start_freq+cumsum( [0, ones(1,num_freq_bands-1)]*one_band_length );
nfft = 2^14; % orig 2^12
window = [];
overlap = [];
resolution = f_s/nfft;
one_band_n_points = floor(one_band_length/resolution);
func = @(x) floor(x/resolution)+1;
band_start_idx = arrayfun(func, band_start_hz);
band_start_idx = [band_start_idx', band_start_idx'+one_band_n_points-1];
channels = [1, 2, 4]; % No Channel 3 
features = zeros(size(segmented_sig, 1), length(channels)*num_freq_bands);
feature_names = strings(0);
psd_avg_feed = zeros(nfft/2 + 1,num_channels); 
psd_avg_base = zeros(nfft/2 + 1,num_channels);
for i = 1:length(segmented_sig)
    tmp = pwelch(segmented_sig{i},[],[],nfft, f_s);
    tmp = tmp ./ sum(tmp,1);
    if labels(i) == 0
        psd_avg_base = psd_avg_base + tmp;
    else
        psd_avg_feed = psd_avg_feed + tmp;
    end
    for ch = 1:length(channels)
       for k = 1:num_freq_bands %10 segments of freq
           features(i, num_freq_bands*(ch-1)+k) = sum(tmp(band_start_idx(k,1):band_start_idx(k,2), channels(ch)));
           if i == 1
               feature_names = [feature_names, strcat("channel ", int2str(channels(ch)),"_",num2str(band_start_hz(k)),"-",num2str(band_start_hz(k)+one_band_length),' Hz')];
           end
       end
    end    
end

psd_avg_base = psd_avg_base / (length(segmented_sig) * num_participants);
psd_avg_feed = psd_avg_feed / (length(segmented_sig) * num_participants);
%% Plot PSD
close all
figure('color','w')
t=tiledlayout(2,3,'TileSpacing','compact','Padding','compact' );
absc = (0:81)*resolution;
% absc = (0:59)*resolution;

for channel = channels
    nexttile();
    plot(absc, psd_avg_base(1:82,channel), 'DisplayName', 'Baseline')
    hold on
    plot(absc, psd_avg_feed(1:82,channel), 'DisplayName', 'Feeding')
    legend()
    title(strcat(['Channel ',int2str(channel)]))
    xlabel('Frequency [Hz]')
end
title(t,"Average normalized PSD for Feeding & Baseline")
%% Classification hyperparameters
rng(1)
% x=zscore(features);
x=features;
numOfFolds = 3;
C = [1,1];
%% forward features selection - Classification using all models
model_names = ["SVM - Linear", "SVM - Polynomial 2", "SVM - Polynomial 3", "SVM - RBF", "Regularized LDA", "QDA", "Logistic Regression", "Random Forest"];

results_forward = struct('fpr',{},'tpr',{},'auc',{},'scores',{},'fs',{});
for i = 1:length(model_names)
    rng(1)
    [fpr, tpr, auc, scores, fs, performance] = classify_with_feature_selection_ROC(x,labels ,numOfFolds , C, model_names(i), "forward");
    results_forward(i).fpr = fpr; results_forward(i).tpr = tpr; results_forward(i).auc = auc; results_forward(i).scores = scores; results_forward(i).fs = fs; results_forward(i).performance = performance;
end

%% backward features selection - Classification using all models
model_names = ["SVM - Linear", "SVM - Polynomial 2", "SVM - Polynomial 3", "SVM - RBF", "Regularized LDA", "QDA", "Logistic Regression", "Random Forest"];

results_backward = struct('fpr',{},'tpr',{},'auc',{},'scores',{},'fs',{});
for i = 1:length(model_names)
    [fpr, tpr, auc, scores, fs, performance] = classify_with_feature_selection_ROC(x,labels ,numOfFolds , C, model_names(i), "backward");
    results_backward(i).fpr = fpr; results_backward(i).tpr = tpr; results_backward(i).auc = auc; results_backward(i).scores = scores; results_backward(i).fs = fs; results_backward(i).performance = performance;
end

save('results_forward_full.mat', 'results_forward')
save('results_backward_full.mat', 'results_backward')
%%
close all
f1=figure('color','w', 'OuterPosition', [5 5 3*800+10 800] );
l1_ypos = 0.47;
l2_ypos = 0.110330469373673;
l1_height=0.5;
l2_height=0.25;
l1_width=0.25;
t1 = subplot(2,3,1,'position',[0.13,l1_ypos,l1_width,l1_height]);
plot(t1, 0:0.5:1,0:0.5:1,'--k', 'LineWidth',1.5,'HandleVisibility','off')
title('Forward Selection')
xlabel('False Positive Rate')
hold on
% f2=figure('color','w', 'OuterPosition', [100 50 800 950]);
t2 = subplot(2,3,2,'position',[0.410797101449275,l1_ypos,l1_width,l1_height]);
plot(t2,0:0.5:1,0:0.5:1,'--k', 'LineWidth',1.5, 'HandleVisibility','off')
title('Backward Selection')
xlabel('False Positive Rate')
hold on
for i = 1:length(results_forward)
    disp_text = model_names(i);% + ", AUC = " + round(results_forward(i).auc, 3);
    plot(t1, results_forward(i).fpr, results_forward(i).tpr, 'DisplayName', disp_text, 'LineWidth',1.5)
    disp_text = model_names(i);% + ", AUC = " + round(results_backward(i).auc, 3);
    plot(t2, results_backward(i).fpr, results_backward(i).tpr, 'DisplayName', disp_text, 'LineWidth',1.5)
end
t3 = subplot(2,3,3,'position',[0.691594202898551,l1_ypos,l1_width,l1_height]);
disp_text = "CNN";% + ", AUC = " + round(results_forward(i).auc, 3);
plot(t3,0:0.5:1,0:0.5:1,'--k', 'LineWidth',1.5, 'HandleVisibility','off')
hold on
plot(t3, fpr, tpr, 'DisplayName', disp_text, 'LineWidth',1.5)
set([t1, t2, t3],'FontSize',14)
legend(t1, 'show','FontSize',11,'location', 'southeast')
legend(t2, 'show','FontSize',11,'location', 'southeast')
legend(t3, 'show','FontSize',11,'location', 'southeast')
xlabel('False Positive Rate')
ylabel(t1,'True Positive Rate')
%
% figure(f1)
subplot(2,3,4, 'position', [0.13,l2_ypos,l1_width,l2_height]);
ax = gca;
bar([results_forward.auc])
set(ax,'xticklabel',model_names,'xticklabelrotation',30)
ylabel('AUC')
% yticks(0.7:0.05:1)
ylim([0.5,1.001])
ax.XAxis.FontSize = 11;
ax.YAxis.FontSize = 14;
% set(ax,'FontSize',18)
subplot(2,3,5,'position',[0.410797101449275,l2_ypos,l1_width,l2_height]);
bar([results_backward.auc])
ax = gca;
% yticks(0.7:0.05:0.9)
ylim([0.5,1.001])
set(ax,'xticklabel',model_names,'xticklabelrotation',30)
ax.XAxis.FontSize = 11;
ax.YAxis.FontSize = 14;

subplot(2,3,6,'position',[0.691594202898551,l2_ypos,l1_width,l2_height]);
bar(categorical(["CNN"]), roc_auc,0.125)
ax = gca;
% yticks(0.7:0.05:0.9)
ylim([0.5,1.001])
ax.XAxis.FontSize = 12;
ax.YAxis.FontSize = 14;
%% Plot AUCs
f_auc=[];
b_auc=[];
f_fs=[];
b_fs=[];
for i= 1 :length(results_forward)
    f_auc=[f_auc, results_forward(i).auc];
    b_auc=[b_auc, results_backward(i).auc];
    f_fs=[f_fs; results_forward(i).fs];
    b_fs=[b_fs; results_backward(i).fs];
end
mean(f_auc);
mean(b_auc);
f_perc = 100*sum(f_fs, 1)/size(f_fs,1);
b_perc = 100*sum(b_fs, 1)/size(b_fs,1);
all_perc = 100*sum([b_fs;f_fs], 1)/(2*size(b_fs,1));
%% Get selected features
forward = {results_forward(:).fs};
backward = {results_backward(:).fs};
forward_fs = zeros(8,30);% (num_models, num_features)
backward_fs = zeros(8,30);
for i = 1:8
     forward_fs(i,:) = forward{i};
     backward_fs(i,:) = backward{i};
end
forward_fs_T = array2table(forward_fs,"VariableNames",feature_names, 'RowNames', model_names);
backward_fs_T = array2table(backward_fs,"VariableNames",feature_names, 'RowNames', model_names);
all_fs = array2table([forward_fs;backward_fs],"VariableNames",feature_names, 'RowNames', ['Forward_'+model_names, 'Backward_'+ model_names]);
freq_forward = array2table(mean(forward_fs)*100,"VariableNames",feature_names, 'RowNames', "Frequency of selection - Forward" );
freq_backward = array2table(mean(backward_fs)*100,"VariableNames",feature_names, 'RowNames', "Frequency of selection - Backward" );
freq_all = array2table(mean([forward_fs;backward_fs])*100,"VariableNames",feature_names, 'RowNames', "Frequency of selection - All" );
T = [all_fs; freq_forward; freq_backward; freq_all];
writetable(T,'feature_selection_results.xlsx','WriteRowNames',true)
%% Get performance per animal
forward_table = get_results_per_subject(results_forward, labels, subject_labels, participants);
backward_table = get_results_per_subject(results_backward, labels, subject_labels, participants);
T = [forward_table; backward_table];
writetable(T,'results_per_animal.xlsx')