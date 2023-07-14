import numpy as np
# from torchsummary import summary
import matplotlib.pyplot as plt
from scipy.io import savemat

# results = {'fpr' : fpr, 'tpr': tpr, 'roc_auc': roc_auc, 'prediction': prediction}
results= np.load('CNN_results.npy',allow_pickle=True).item()
labels = np.load('CNN_labels.npy',allow_pickle=True).item()
subjects = ['103-21', '58-21', '60-21', '87-21']
results ['prediction'] = results['prediction'].numpy()
res = results['prediction'] >=0.5
acc = np.zeros_like(subjects)
sens = np.zeros_like(subjects)
spec = np.zeros_like(subjects)
for i in range(len(subjects)):
    idx = labels['subject_labels'] == subjects[i]
    tmp_res = res[idx]
    tmp_labels = labels['labels'][idx]
    acc[i] = 100*(np.sum(tmp_res == tmp_labels)/np.sum(idx))
    base = tmp_labels == 0
    feed = tmp_labels == 1
    spec[i] = 100*(np.sum(tmp_res[base] == tmp_labels[base])/np.sum(base))
    sens[i] = 100*(np.sum(tmp_res[feed] == tmp_labels[feed])/np.sum(feed))

savemat('cnn_results.mat', results)
font = {#'family' : 'Helvetica',
        # 'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)
plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
lw = 1.5
plt.plot(
    results['fpr'],
    results['tpr'],
    color="darkorange",
    lw=lw,
    label="CNN, AUC = %0.3f)" % results['roc_auc'],
)
plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
plt.xlim([-0.003, 1.0])
plt.ylim([0.0, 1.003])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right", fontsize = 14)
plt.show()
# plt.savefig('Figs/ROC_AUC_CNN.png', dpi=800)

print('all')