import sys
sys.path.append('../../')
import argparse
import pandas as pd
import torch
from utils import datasetSplit, DatasetTorch
from torch.utils.data import DataLoader
from utils import netModels
import copy


def main(datasetName, encoding, filterbank, channels, bins, structure):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##### Dataset load #####
    sourceFolder = '../../datasets/HumanActivityRecognition/datasetSonograms/'
    fileName = f'{sourceFolder}sonograms_{datasetName}{filterbank}{channels}x{bins}{encoding}.bin'
    trainSource, trainTarget, testSource, testTarget, numClass = datasetSplit(fileName, 'CNN')

    trainSet = DatasetTorch(trainSource, trainTarget, numClass, device)
    trainSetDL = DataLoader(trainSet, batch_size=1)
    testSet = DatasetTorch(testSource, testTarget, numClass, device)

    ##### Model definition #####
    dataShape = trainSource.shape
    modelCNN = netModels(dataShape, structure, numClass).to(device)
    optimizer = torch.optim.Adam(modelCNN.parameters(), lr=0.001)

    ##### Training loop #####
    running = []
    for _ in range(60):
        modelCNN.train()
        for source, target in trainSetDL:
            optimizer.zero_grad()
            loss = torch.nn.CrossEntropyLoss()(modelCNN(source), target)
            loss.backward()
            optimizer.step()

        modelCNN.eval()
        with torch.no_grad():
            targetTrue = trainSet.target.argmax(axis=1)
            targetPred = modelCNN(trainSet.source).argmax(axis=1)
            accuracyTrain = torch.sum(targetPred == targetTrue).item()/targetTrue.shape[0]

            targetTrue = testSet.target.argmax(axis=1)
            targetPred = modelCNN(testSet.source).argmax(axis=1)
            accuracyTest = torch.sum(targetPred == targetTrue).item()/targetTrue.shape[0]

        running.append([accuracyTrain, accuracyTest, copy.deepcopy(modelCNN)])

    return sorted(running, key=lambda x: (x[1], x[0]), reverse=True)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-n', '--datasetName', help='Dataset file name', type=str, default='subset1')
    parser.add_argument('-e', '--encoding', help='Encoding algorithm selected', type=str, default='RATE')
    parser.add_argument('-f', '--filterbank', help='Type of filterbank', type=str, default='butterworth')
    parser.add_argument('-c', '--channels', help='Frequency decomposition channels', type=int, default=4)
    parser.add_argument('-b', '--bins', help='Binning width', type=int, default=24)
    parser.add_argument('-s', '--structure', help='Network structure', type=str, default='c06c12f2')
    parser.add_argument('-t', '--trials', help='Trials training', type=int, default=30)

    argument = parser.parse_args()

    datasetName = argument.datasetName
    encoding = argument.encoding
    filterbank = argument.filterbank
    channels = argument.channels
    bins = argument.bins
    structure = argument.structure
    trials = argument.trials

    ##### Verify stored model #####
    columnLabels = ['filterbank', 'channels', 'bins', 'encoding', 'structure', 'train', 'test']
    flagCompute = True
    sourceFolder = '../../networkPerformance/HumanActivityRecognition/'
    fileName = f'{sourceFolder}{datasetName}CNN-ModelComplete.csv'
    try:
        performanceData = pd.read_csv(fileName)
        flagCompute = not bool(len(performanceData[
            (performanceData['encoding'] == encoding) &
            (performanceData['filterbank'] == filterbank) &
            (performanceData['channels'] == channels) &
            (performanceData['bins'] == bins) &
            (performanceData['structure'] == structure)
        ]))
    except:
        pass

    ##### Training models #####
    # print(datasetName, encoding, filterbank, channels, bins, structure)
    if flagCompute == True:
        history = []
        for trial in range(trials):
            accuracyTrain, accuracyTest, modelCNN = main(datasetName, encoding, filterbank, channels, bins, structure)
            history.append((accuracyTrain, accuracyTest, modelCNN))

        accuracyTrain, accuracyTest, modelCNN = sorted(history, key=lambda x: (x[1], x[0]), reverse=True)[0]

        ##### Save model #####
        sourceFolder = '../../networkModels/HumanActivityRecognition/complete/'
        torch.save(modelCNN, f'{sourceFolder}{datasetName}{filterbank}{channels}x{bins}{structure}{encoding}.pth')

        ##### Save performance #####
        try:
            performanceData = pd.read_csv(fileName)
            performanceData = performanceData.values.tolist()
            performanceData.append([filterbank, channels, bins, encoding, structure, accuracyTrain, accuracyTest])
            performanceData = pd.DataFrame(performanceData, index=None, columns=columnLabels)
            performanceData.to_csv(fileName, index=False)
        except:
            performanceData = [[filterbank, channels, bins, encoding, structure, accuracyTrain, accuracyTest]]
            performanceData = pd.DataFrame(performanceData, index=None, columns=columnLabels)
            performanceData.to_csv(fileName, index=False)
