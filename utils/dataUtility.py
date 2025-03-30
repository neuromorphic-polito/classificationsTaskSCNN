import random
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch


def datasetSplit(fileName, netType):
    ##### Seed reproducibility #####
    random.seed(0)

    ##### Load data #####
    file = open(fileName, 'rb')
    datasetSonograms = pickle.load(file)
    file.close()

    numSample, numClass = len(datasetSonograms[0]), len(datasetSonograms.keys())
    splitTrain = int(np.floor(numSample*0.8))

    ##### Data preprocessing #####
    trainSet = [(key, value) for key in datasetSonograms.keys() for value in datasetSonograms[key][0:splitTrain]]
    testSet = [(key, value) for key in datasetSonograms.keys() for value in datasetSonograms[key][splitTrain:]]

    random.shuffle(trainSet)
    random.shuffle(testSet)

    trainSource = np.vstack([[value] for _, value in trainSet])
    trainTarget = np.vstack([label for label, _ in trainSet])

    testSource = np.vstack([[value] for _, value in testSet])
    testTarget = np.vstack([label for label, _ in testSet])

    trainSource /= trainSource.max()
    testSource /= testSource.max()

    trainTarget = trainTarget.flatten()
    testTarget = testTarget.flatten()

    if netType == 'CNN':
        trainSource = np.expand_dims(trainSource, axis=1)
        testSource = np.expand_dims(testSource, axis=1)
    elif netType == 'SNN':
        trainSource *= 255.0
        testSource *= 255.0
    else:
        raise Exception(f'Network {netType} not available')

    return trainSource, trainTarget, testSource, testTarget, numClass


class DatasetTorch(Dataset):
    def __init__(self, source, target, numClass, device):
        self.source = torch.from_numpy(source).float().to(device)

        self.target = torch.from_numpy(target).to(device)
        self.target = torch.nn.functional.one_hot(self.target, num_classes=numClass).to(float)

    def __len__(self):
        return self.source.shape[0]

    def __getitem__(self, item):
        return self.source[item], self.target[item]
