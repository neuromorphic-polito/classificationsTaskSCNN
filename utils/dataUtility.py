import random
import pickle
import numpy as np

def datasetSplitting(fileName, netType):
    random.seed(0)

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

    if netType == 'CNN':
        trainSource = np.expand_dims(trainSource, axis=3)
        testSource = np.expand_dims(testSource, axis=3)
    elif netType == 'SNN':
        trainSource *= 255.0
        testSource *= 255.0

        trainTarget = trainTarget.flatten()
        testTarget = testTarget.flatten()
    else:
        raise Exception(f'Network {netType} not available')

    return trainSource, trainTarget, testSource, testTarget, numClass
