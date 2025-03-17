import sys
sys.path.append('../../')
import argparse
import pandas as pd
from utils import datasetSplitting
from utils import netModelsComplete


#############################
# ##### Training loop ##### #
#############################
def main(encoding, filterbank, channels, bins, structure):
    ##### Dataset loading #####
    sourceFolder = '../../datasets/FreeSpokenDigits/datasetSonograms/'
    fileName = f'{sourceFolder}sonograms_{filterbank}{channels}x{bins}{encoding}.bin'
    trainSource, trainTarget, testSource, testTarget, numClass = datasetSplitting(fileName, 'CNN')

    ##### Model definitions #####
    dataShape = trainSource.shape[1:]
    modelCNN = netModelsComplete(structure, dataShape, numClass)

    modelCNN.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    running = []
    for _ in range(60):
        accuracyTrain = modelCNN.fit(x=trainSource, y=trainTarget, epochs=1, batch_size=1, verbose=0)
        accuracyTest = modelCNN.evaluate(x=testSource, y=testTarget, verbose=0)
        running.append([accuracyTrain.history['accuracy'][-1], accuracyTest[-1], modelCNN])

    return sorted(running, key=lambda x: (x[1], x[0]), reverse=True)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-e', '--encoding', help='Encoding algorithm selected', type=str, default='RATE')
    parser.add_argument('-f', '--filterbank', help='Type of filterbank', type=str, default='butterworth')
    parser.add_argument('-c', '--channels', help='Frequency decomposition channels', type=int, default=32)
    parser.add_argument('-b', '--bins', help='Binning width', type=int, default=50)
    parser.add_argument('-s', '--structure', help='Network structure', type=str, default='c06c12f2')
    parser.add_argument('-t', '--trials', help='Trials training', type=int, default=30)

    argument = parser.parse_args()

    encoding = argument.encoding
    filterbank = argument.filterbank
    channels = argument.channels
    bins = argument.bins
    structure = argument.structure
    trials = argument.trials

    ##### Check model already calculated #####
    columnLabels = ['Filterbank', 'Channels', 'Bins', 'Encoding', 'Structure', 'Train', 'Test']
    flagCompute = True
    sourceFolder = '../../networkPerformance/FreeSpokenDigits/'
    fileName = f'{sourceFolder}CNN-ModelComplete.csv'
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
    print(encoding, filterbank, channels, bins, structure)

    if flagCompute == True:
        history = []
        for trial in range(trials):
            accuracyTrain, accuracyTest, modelCNN = main(encoding, filterbank, channels, bins, structure)
            history.append((accuracyTrain, accuracyTest, modelCNN))

        accuracyTrain, accuracyTest, modelCNN = sorted(history, key=lambda x: (x[1], x[0]), reverse=True)[0]

        ##### Save data of models #####
        sourceFolder = '../../networkModels/FreeSpokenDigits/complete/'
        modelCNN.save(f'{sourceFolder}{filterbank}{channels}x{bins}{structure}{encoding}.keras')

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
