import os
import pandas as pd
import numpy as np


########################
# ##### Accuracy ##### #
########################

os.system('rm ../../networkPerformance/FreeSpokenDigits/accuracy.csv')

performanceCNN = pd.read_csv('../../networkPerformance/FreeSpokenDigits/CNN-ModelComplete.csv')
performanceSNN = pd.read_csv('../../networkPerformance/FreeSpokenDigits/SCNN-ModelCompleteReduced.csv')

structures = ['c06c12f2', 'c12c24f2']
for structure in structures:
    accuracyTrainCNN = pd.pivot_table(
        performanceCNN[performanceCNN['structure'] == structure],
        index=['structure', 'filterbank', 'channels', 'bins'], columns='encoding',
        values='train',
    )*100
    accuracyTestCNN = pd.pivot_table(
        performanceCNN[performanceCNN['structure'] == structure],
        index=['structure', 'filterbank', 'channels', 'bins'], columns='encoding',
        values='test',
    )*100
    accuracyTrainSNN = pd.pivot_table(
        performanceSNN[(performanceSNN['structure'] == structure) & (performanceSNN['reduction'] == 0)],
        index=['structure', 'filterbank', 'channels', 'bins'], columns='encoding',
        values='train',
    )*100
    accuracyTestSNN = pd.pivot_table(
        performanceSNN[(performanceSNN['structure'] == structure) & (performanceSNN['reduction'] == 0)],
        index=['structure', 'filterbank', 'channels', 'bins'], columns='encoding',
        values='test',
    )*100

    encodings = ['RATE', 'TBR', 'SF', 'ZCSF', 'MW', 'HSA', 'MHSA', 'BSA', 'PHASE', 'TTFS', 'BURST']
    accuracyTrainCNN = accuracyTrainCNN.reindex(encodings, axis=1)
    accuracyTestCNN = accuracyTestCNN.reindex(encodings, axis=1)
    accuracyTrainSNN = accuracyTrainSNN.reindex(encodings, axis=1)
    accuracyTestSNN = accuracyTestSNN.reindex(encodings, axis=1)

    index = accuracyTestCNN.index
    columns = accuracyTestCNN.columns

    accuracyTrainCNN = accuracyTrainCNN.to_numpy()
    accuracyTestCNN = accuracyTestCNN.to_numpy()
    accuracyTrainSNN = accuracyTrainSNN.to_numpy()
    accuracyTestSNN = accuracyTestSNN.to_numpy()

    accuracyCNN = []
    for i in range(accuracyTrainCNN.shape[1]):
        accuracyCNN.append(accuracyTrainCNN[:, i])
        accuracyCNN.append(accuracyTestCNN[:, i])
    accuracyCNN = np.vstack(accuracyCNN).T
    accuracyCNN = pd.DataFrame(accuracyCNN, index=index, columns=[f'CNN_{e}_{t}' for e in encodings for t in ['train', 'test']])

    accuracySNN = []
    for i in range(accuracyTrainSNN.shape[1]):
        accuracySNN.append(accuracyTrainSNN[:, i])
        accuracySNN.append(accuracyTestSNN[:, i])
    accuracySNN = np.vstack(accuracySNN).T
    accuracySNN = pd.DataFrame(accuracySNN, index=index, columns=[f'SNN_{e}_{t}' for e in encodings for t in ['train', 'test']])

    accuracyCNN.to_csv('../../networkPerformance/FreeSpokenDigits/accuracy.csv', mode='a')
    emptyLine = pd.DataFrame([np.nan]*2)
    emptyLine.to_csv('../../networkPerformance/FreeSpokenDigits/accuracy.csv', mode='a', index=False, columns=[])
    accuracySNN.to_csv('../../networkPerformance/FreeSpokenDigits/accuracy.csv', mode='a')
    emptyLine = pd.DataFrame([np.nan]*2)
    emptyLine.to_csv('../../networkPerformance/FreeSpokenDigits/accuracy.csv', mode='a', index=False, columns=[])
