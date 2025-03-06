import sys
sys.path.append('../../')
import argparse
from utils import datasetSplitting
from utils import netModelsComplete
import numpy as np
import pandas as pd


#############################
# ##### Training loop ##### #
#############################
def main(datasetName, encoding, filterbank, channels, bins, structure):
    ##### Dataset loading #####
    sourceFolder = '../../datasets/HumanActivityRecognition/datasetSonograms/'
    fileName = f'{sourceFolder}sonogram_{datasetName}{filterbank}{channels}x{bins}{encoding}.bin'
    trainSource, trainTarget, testSource, testTarget, numClass = datasetSplitting(fileName, 'CNN')

    ##### Model definitions #####
    dataShape = trainSource.shape[1:]
    modelCNN = netModelsComplete(structure, dataShape, numClass)

    modelCNN.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    runningTrain, runningTest, runningModel = [], [], []
    for epochs in range(60):
        accuracyTrain = modelCNN.fit(x=trainSource, y=trainTarget, validation_split=0.1, epochs=1, batch_size=1, verbose=0)
        accuracyTest = modelCNN.evaluate(x=testSource, y=testTarget, verbose=0)
        runningTrain.append(accuracyTrain.history['accuracy'][-1])
        runningTest.append(accuracyTest[-1])
        runningModel.append(modelCNN)

    selected = np.argmax(runningTest)

    return runningTrain[selected], runningTest[selected], runningModel[selected]


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

    ##### Check model already calculated #####
    columnLabels = ['Filterbank', 'Channels', 'Bins', 'Encoding', 'Structure', 'Train', 'Test']
    flagCompute = True
    sourceFolder = '../../networkPerformance/HumanActivityRecognition/'
    fileName = f'{sourceFolder}{datasetName}CNN-ModelComplete.csv'
    try:
        performanceData = pd.read_csv(fileName)
        flagCompute = not bool(len(performanceData[
            (performanceData['Encoding'] == encoding) &
            (performanceData['Filterbank'] == filterbank) &
            (performanceData['Channels'] == channels) &
            (performanceData['Bins'] == bins) &
            (performanceData['Structure'] == structure)
        ]))
    except:
        pass

    ##### Run training models #####
    print(datasetName, encoding, filterbank, channels, bins, structure)

    if flagCompute == True:
        metrics = np.zeros(trials)
        history = []
        for trial in range(trials):
            accuracyTrain, accuracyTest, modelCNN = main(datasetName, encoding, filterbank, channels, bins, structure)
            history.append((accuracyTrain, accuracyTest, modelCNN))
            metrics[trial] = (1-accuracyTrain)**2+(1-accuracyTest)**2

        accuracyTrain, accuracyTest, modelCNN = history[np.argmin(metrics)]

        ##### Save data of models #####
        sourceFolder = '../../networkModels/HumanActivityRecognition/complete/'
        modelCNN.save(f'{sourceFolder}{datasetName}{filterbank}{channels}x{bins}{structure}{encoding}.keras')

        ##### Save data for performance #####
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
