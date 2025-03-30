import sys
sys.path.append('../../')
import argparse
import pandas as pd
import torch
from utils import datasetSplit, DatasetTorch
from torch.utils.data import DataLoader
from utils import netModels
import copy


def main(encoding, filterbank, channels, bins, structure, quartile):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##### Dataset load #####
    sourceFolder = '../../datasets/FreeSpokenDigits/datasetSonograms/'
    fileName = f'{sourceFolder}sonograms_{filterbank}{channels}x{bins}{encoding}.bin'
    trainSource, trainTarget, testSource, testTarget, numClass = datasetSplit(fileName, 'CNN')

    trainSet = DatasetTorch(trainSource, trainTarget, numClass, device)
    trainSetDL = DataLoader(trainSet, batch_size=1)
    testSet = DatasetTorch(testSource, testTarget, numClass, device)

    ##### Load model #####
    sourceFolder = '../../networkModels/FreeSpokenDigits/complete/'
    modelCNN = torch.load(f'{sourceFolder}{filterbank}{channels}x{bins}{structure}{encoding}.pth', weights_only=False)

    ##### Quartile calculation #####
    layersWeigths = modelCNN.state_dict()
    masks = []
    for key in layersWeigths:
        threshold = None
        if quartile == 'median':
            threshold = torch.quantile(torch.abs(layersWeigths[key].flatten()), 0.5)
        elif quartile == 'upper':
            threshold = torch.quantile(torch.abs(layersWeigths[key].flatten()), 0.75)
        mask = torch.where(torch.abs(layersWeigths[key]) < threshold, 0.0, 1.0)
        masks.append(mask)
        layersWeigths[key] *= mask
    synapses = torch.sum(torch.hstack([m.flatten() for m in masks]), dtype=int).detach().cpu().item()

    ##### Model definition #####
    dataShape = trainSource.shape
    modelCNNPruned = netModels(dataShape, structure, numClass).to(device)
    modelCNNPruned.load_state_dict(layersWeigths)
    optimizer = torch.optim.Adam(modelCNNPruned.parameters(), lr=0.001)

    ##### Training loop #####
    running = []
    for _ in range(20):
        modelCNNPruned.train()
        for source, target in trainSetDL:
            optimizer.zero_grad()
            loss = torch.nn.CrossEntropyLoss()(modelCNNPruned(source), target)
            loss.backward()
            optimizer.step()

            layersWeigths = modelCNNPruned.state_dict()
            for i, key in enumerate(layersWeigths):
                layersWeigths[key] *= masks[i]
            modelCNNPruned.load_state_dict(layersWeigths)

        modelCNNPruned.eval()
        with torch.no_grad():
            targetTrue = trainSet.target.argmax(axis=1)
            targetPred = modelCNNPruned(trainSet.source).argmax(axis=1)
            accuracyTrain = torch.sum(targetPred == targetTrue).item()/targetTrue.shape[0]

            targetTrue = testSet.target.argmax(axis=1)
            targetPred = modelCNNPruned(testSet.source).argmax(axis=1)
            accuracyTest = torch.sum(targetPred == targetTrue).item()/targetTrue.shape[0]

        running.append([accuracyTrain, accuracyTest, synapses, copy.deepcopy(modelCNNPruned)])

    return sorted(running, key=lambda x: (x[1], x[0], x[2]), reverse=True)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-e', '--encoding', help='Encoding algorithm selected', type=str, default='RATE')
    parser.add_argument('-f', '--filterbank', help='Type of filterbank', type=str, default='butterworth')
    parser.add_argument('-c', '--channels', help='Frequency decomposition channels', type=int, default=32)
    parser.add_argument('-b', '--bins', help='Binning width', type=int, default=50)
    parser.add_argument('-s', '--structure', help='Network structure', type=str, default='c06c12f2')
    parser.add_argument('-q', '--quartile', help='Quartile pruning', type=str, default='median')

    argument = parser.parse_args()

    encoding = argument.encoding
    filterbank = argument.filterbank
    channels = argument.channels
    bins = argument.bins
    structure = argument.structure
    quartile = argument.quartile

    ##### Verify stored model #####
    columnLabels = ['Filterbank', 'Channels', 'Bins', 'Encoding', 'Structure', 'Quartile', 'Synapses', 'Train', 'Test']
    flagCompute = True
    sourceFolder = '../../networkPerformance/FreeSpokenDigits/'
    fileName = f'{sourceFolder}CNN-ModelPruned.csv'
    try:
        performanceData = pd.read_csv(fileName)
        flagCompute = not bool(len(performanceData[
            (performanceData['Encoding'] == encoding) &
            (performanceData['Filterbank'] == filterbank) &
            (performanceData['Channels'] == channels) &
            (performanceData['Bins'] == bins) &
            (performanceData['Structure'] == structure) &
            (performanceData['Quartile'] == quartile)
        ]))
    except:
        pass

    ##### Training models #####
    # print(encoding, filterbank, channels, bins, structure, quartile)
    if flagCompute == True:
        accuracyTrain, accuracyTest, synapses, modelCNNPruned = main(encoding, filterbank, channels, bins, structure, quartile)

        ##### Save model #####
        sourceFolder = '../../networkModels/FreeSpokenDigits/pruned/'
        torch.save(modelCNNPruned, f'{sourceFolder}{filterbank}{channels}x{bins}{structure}{quartile}{encoding}.pth')

        ##### Save performance #####
        try:
            performanceData = pd.read_csv(fileName)
            performanceData = performanceData.values.tolist()
            performanceData.append([filterbank, channels, bins, encoding, structure, quartile, synapses, accuracyTrain, accuracyTest])
            performanceData = pd.DataFrame(performanceData, index=None, columns=columnLabels)
            performanceData.to_csv(fileName, index=False)
        except:
            performanceData = [[filterbank, channels, bins, encoding, structure, quartile, synapses, accuracyTrain, accuracyTest]]
            performanceData = pd.DataFrame(performanceData, index=None, columns=columnLabels)
            performanceData.to_csv(fileName, index=False)
