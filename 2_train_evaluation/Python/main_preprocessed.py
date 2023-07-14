import copy
import numpy as np
from read_preprocessed import *
import torch
import torch.nn as nn
import models_1D
from tqdm import trange
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Read data
data_fname  ="../preprocessed data"
data_loaded = False # Load data from npy file(True) or read csv(False)
subjects = ['103-21', '87-21', '58-21', '60-21'] # subjects = 'all'
if not data_loaded:
    d = data(data_fname, subjects=subjects)
    np.save('data_object.npy', d)
else:
    d = np.load('data_object.npy', allow_pickle=True).item()

# segment 1 hour data into 1 minute chunks - total = 60 segments
segment_length = 1*12000 # 12000 points = 60 sec
d.segment(segment_length)
for i in d.used_subjects_names:
    print('{}:\nN baseline: {} N feeding: {}'.format(i,len(d.data[i].baseline), len(d.data[i].feeding)))
data_, labels, subject_label = d.prepare_sets(used_segments=[0,60]) #choose used_segments from [0:60] NOTE second number not inculded
CNN_labels = {'labels': labels, 'subject_labels': subject_label, 'subjects_ordered': ['103-21', '87-21', '58-21', '60-21']}
np.save('CNN_labels.npy', CNN_labels)

# Training hyperparameters
LR = 0.001
EPOCHS = 200 # 30
BATCH_SIZE = 20 #4 #20
patience = 50 # for early stopping
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.manual_seed(1)
np.random.seed(1)
# used channels
channels = [0,1,3,]
num_input_channels = len(channels)
# 3 fold cross validation
k_folds = 3
dataset = torch.utils.data.TensorDataset(torch.tensor(data_[:,channels]).float(), torch.tensor(labels).float())
kfold = KFold(n_splits=k_folds, shuffle=True)

t1 = trange(0, k_folds) # Folds
t2 = trange(0, EPOCHS) # Epochs
t3 = trange(0, int(320/BATCH_SIZE)) # Batches

accuracy = np.zeros((k_folds))
sensitivity = np.zeros((k_folds))
specificity = np.zeros((k_folds))
prediction = torch.zeros((len(dataset),1))

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset[:][0])):

    t2.reset()
    train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
    train = torch.utils.data.DataLoader(
        dataset = dataset, # [indices[mask].flatten()],  # torch TensorDataset format
        batch_size = BATCH_SIZE,  # mini batch size
        #num_workers=1,
        #pin_memory=True,
        sampler=train_sampler,
    )
    test = dataset[test_ids]
    c = models_1D.cnn(num_input_channels ,segment_length, device).to(device)
    optimizer = torch.optim.AdamW(c.parameters(), lr=LR)
    lossfcn = nn.BCELoss().to(device)
    best_loss = np.Inf
    counter = 0 # for early stopping
    for epoch in range(EPOCHS):
        t3.reset()
        c.train()
        for step, (x, y) in enumerate(train):
            x, y= x.to(device), y.to(device)
            c.zero_grad()
            out = c(x)
            loss = lossfcn(out, y)
            loss.backward()
            optimizer.step()
            t3.set_description('Epoch: %04d | num_batch: %04d | Batch loss: %.5f ' % (epoch, step, loss.data.item()))
            t3.update(1)
        c.eval()
        with torch.no_grad():
            out_test = c(test[:][0].to(device))
            loss_ = lossfcn(out_test, test[:][1].to(device)).cpu().item()

        if loss_ < best_loss:
            best_model = copy.deepcopy(c)
            best_loss = loss_
            counter = 0
        else:
            counter += 1
        # Early stopping
        if counter > patience: 
            break
        t2.set_description('Epoch: %04d | Test loss: %.5f ' % (epoch, loss_))
        t2.update(1)

    # torch.cuda.empty_cache()
    best_model.eval()
    with torch.no_grad():
        out_test = best_model(test[0].to(device)).float().cpu()
    prediction[test_ids] = out_test
    
    res_predict = out_test > 0.5
    acc = 100*(torch.sum(res_predict == test[1])/len(test_ids)).numpy()
    baseline_idx = (test[:][1] == 0).squeeze()
    feeding_idx= (test[:][1]== 1).squeeze()
    spec = 100*(torch.sum(res_predict[baseline_idx] == test[1][baseline_idx])/sum(baseline_idx)).numpy()
    sens = 100*(torch.sum(res_predict[feeding_idx] == test[1][feeding_idx])/sum(feeding_idx)).numpy()
    accuracy[fold] = acc
    sensitivity[fold] = sens
    specificity[fold] = spec
    t1.set_description('accuracy fold %d: %.2f, - stop epoch : %d '%(fold, acc, epoch))
    t1.update(1)

t1.close()
t2.close()
t3.close()

fpr, tpr, _ = roc_curve(dataset[:][1], prediction, drop_intermediate=False)
roc_auc = auc(fpr, tpr)
results = {'fpr' : fpr, 'tpr': tpr, 'roc_auc': roc_auc, 'prediction': prediction}
np.save('CNN_results.npy', results)

font = {#'family' : 'Helvetica',
        # 'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)
plt.figure(figsize=(8,8))
lw = 1.5
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="CNN, AUC = %0.3f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
plt.xlim([-0.003, 1.0])
plt.ylim([0.0, 1.003])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right", fontsize = 14)
plt.show()
plt.savefig('Figs/ROC_AUC_CNN.png', dpi=800)

print('all')