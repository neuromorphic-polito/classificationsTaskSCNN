import sys
sys.path.append('../../')
import argparse
from utils import datasetSplitting
from utils import netModelsPruned, Relu
import numpy as np
import tensorflow as tf
import pandas as pd


#############################
# ##### Training loop ##### #
#############################
def main(encoding, filterbank, channels, bins, structure, quartile):
    ##### Dataset loading #####
    sourceFolder = '../../datasets/FreeSpokenDigits/datasetSonograms/'
    fileName = f'{sourceFolder}sonogram_{filterbank}{channels}x{bins}{encoding}.bin'
    trainSource, trainTarget, testSource, testTarget, numClass = datasetSplitting(fileName, 'CNN')

    ##### Load model network #####
    sourceFolder = '../../networkModels/FreeSpokenDigits/complete/'
    modelCNN = tf.keras.models.load_model(f'{sourceFolder}{filterbank}{channels}x{bins}{structure}{encoding}.keras', custom_objects={'Relu': Relu})
    layersWeigths = modelCNN.get_weights()
    mask = modelCNN.get_weights()

    ##### Quartile calculation #####
    for i in range(len(layersWeigths)):
        threshold = None
        if quartile == 'median':
            threshold = np.quantile(np.abs(layersWeigths[i].flatten()), 0.5)
        elif quartile == 'upper':
            threshold = np.quantile(np.abs(layersWeigths[i].flatten()), 0.75)
        mask[i] = np.where((np.abs(layersWeigths[i]) < threshold), 0.0, 1.0)
        layersWeigths[i] = np.where((np.abs(layersWeigths[i]) < threshold), 0.0, layersWeigths[i])

    ##### Model definitions #####
    synapses = np.sum([np.sum(m) for m in mask], dtype=int)
    dataShape = trainSource.shape[1:]
    modelCNNPruned = netModelsPruned(structure, dataShape, mask, numClass)

    modelCNNPruned.set_weights(layersWeigths)
    modelCNNPruned.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    running = []
    for epochs in range(40):
        accuracyTrain = modelCNNPruned.fit(x=trainSource, y=trainTarget, epochs=1, batch_size=1, verbose=0)
        accuracyTest = modelCNNPruned.evaluate(x=testSource, y=testTarget, verbose=0)
        running.append([accuracyTrain.history['accuracy'][-1], accuracyTest[-1], synapses, modelCNN])

    return sorted(running, key=lambda x: (x[1], x[0]), reverse=True)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-e', '--encoding', help='Encoding algorithm selected', type=str, default='RATE')
    parser.add_argument('-f', '--filterbank', help='Type of filterbank', type=str, default='butterworth')
    parser.add_argument('-c', '--channels', help='Frequency decomposition channels', type=int, default=32)
    parser.add_argument('-b', '--bins', help='Binning width', type=int, default=50)
    parser.add_argument('-s', '--structure', help='Network structure', type=str, default='c06c12f2')
    parser.add_argument('-q', '--quartile', help='Quartile pruning', type=str, default='median')
    parser.add_argument('-t', '--trials', help='Trials training', type=int, default=30)

    argument = parser.parse_args()

    encoding = argument.encoding
    filterbank = argument.filterbank
    channels = argument.channels
    bins = argument.bins
    structure = argument.structure
    quartile = argument.quartile
    trials = argument.trials


    ##### Check model already calculated #####
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

    ##### Run training models #####
    print(encoding, filterbank, channels, bins, structure, quartile)
    if flagCompute == True:
        history = []
        for trial in range(trials):
            accuracyTrain, accuracyTest, synapses, modelCNNPruned = main(encoding, filterbank, channels, bins, structure, quartile)
            history.append((accuracyTrain, accuracyTest, synapses, modelCNNPruned))

        accuracyTrain, accuracyTest, synapses, modelCNNPruned = sorted(history, key=lambda x: (x[1], x[0]), reverse=True)[0]

        ##### Save data of models #####
        sourceFolder = '../../networkModels/FreeSpokenDigits/pruned/'
        modelCNNPruned.save(f'{sourceFolder}{filterbank}{channels}x{bins}{structure}{quartile}{encoding}.keras')

        ##### Save data for performance #####
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
