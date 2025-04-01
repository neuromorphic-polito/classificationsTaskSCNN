import pandas as pd

accuracy = pd.read_csv('../../networkPerformance/FreeSpokenDigits/CNN-ModelComplete.csv')


encodings = ['RATE', 'TBR', 'SF', 'ZCSF', 'MW', 'HSA', 'MHSA', 'BSA', 'PHASE', 'TTFS', 'BURST']
successCount = pd.pivot_table(
    accuracy,   # [accuracy['structure'] == 'c06c12f2'],
    index=['structure', 'filterbank', 'channels', 'bins'], columns='encoding',
    values='train',
)*100

successCount = successCount.reindex(encodings, axis=1)

print(successCount)

print()