import sys
sys.path.append('../../')
import argparse
from utils import datasetSplitting
from utils import Dataset
import tensorflow as tf
from utils import CNN, SNN, Relu, Masking
import numpy as np
import pandas as pd


def spikeLabeling(datasetClass, trueLabel, timeSimulation, spikeTrainNest):
    ##### Conversion NEO to Numpy format #####
    numberSamples = trueLabel.size
    spikeOutput = [np.array(neuron) for neuron in spikeTrainNest[-1]]

    ##### Accuracy #####
    spikeCount = np.zeros((datasetClass, numberSamples), dtype=int)
    bins = np.linspace(0, timeSimulation*numberSamples, numberSamples+1)
    for i in range(datasetClass):
        for v in np.searchsorted(bins, spikeOutput[i])-1:
            spikeCount[i, v] += 1
    prediction = np.argmax(spikeCount, axis=0)
    accuracy = np.sum(prediction == trueLabel)/numberSamples

    return accuracy


##############################
# ##### Inference loop ##### #
##############################
def main(encoding, filterbank, channels, bins, structure, quartile):
    ###############################
    # ##### Dataset loading ##### #
    ###############################
    sourceFolder = '../../datasets/FreeSpokenDigits/datasetSonograms/'
    fileName = f'{sourceFolder}sonograms_{filterbank}{channels}x{bins}{encoding}.bin'
    trainSource, trainTarget, testSource, testTarget, numClass = datasetSplitting(fileName, 'SNN')

    timeStimulus = {'duration': 1000.0, 'silence': 30.0}

    dataset = Dataset(
        {'trainSet': (trainSource, trainTarget), 'testSet': (testSource, testTarget)},
        timeStimulus, 'poisson'
    )

    ##########################################
    # ##### Load pre-trained CNN model ##### #
    ##########################################
    ##### Weigths load #####
    sourceFolder = '../../networkModels/FreeSpokenDigits/pruned/'
    modelCNN = tf.keras.models.load_model(f'{sourceFolder}{filterbank}{channels}x{bins}{structure}{quartile}{encoding}.keras', custom_objects={'Relu': Relu, 'Masking': Masking})

    modelCNN = CNN(modelCNN)

    ###################################
    # ##### Simulation with SNN ##### #
    ###################################
    ##### Neuron parameter #####
    lifParams = {
        'cm': 0.25,  # nF
        'i_offset': 0.1,  # nA
        'tau_m': 20.0,  # ms
        'tau_refrac': 1.0,  # ms
        'tau_syn_E': 5.0,  # ms
        'tau_syn_I': 5.0,  # ms
        'v_reset': -65.0,  # mV
        'v_rest': -65.0,  # mV
        'v_thresh': -50.0  # mV
    }
    timeSimulation = timeStimulus['duration']+timeStimulus['silence']

    modelSNN = SNN(dataset.shape, dataset.trainSetSpike, modelCNN, lifParams)

    spikeTrainNest = modelSNN.start_simulation(len(dataset.trainSet), timeStimulus)
    accuracyTrain = spikeLabeling(numClass, trainTarget, timeSimulation, spikeTrainNest)

    modelSNN = SNN(dataset.shape, dataset.testSetSpike, modelCNN, lifParams)

    spikeTestNest = modelSNN.start_simulation(len(dataset.testSet), timeStimulus)
    accuracyTest = spikeLabeling(numClass, testTarget, timeSimulation, spikeTestNest)

    return accuracyTrain, accuracyTest, modelSNN.synapsesNum


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

    ##### Check model already calculated #####
    columnLabels = ['Filterbank', 'Channels', 'Bins', 'Encoding', 'Structure', 'Quartile', 'Synapses', 'Train', 'Test']
    flagCompute = True
    sourceFolder = '../../networkPerformance/FreeSpokenDigits/'
    fileName = f'{sourceFolder}SCNN-ModelPruned.csv'
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

    print(encoding, filterbank, channels, bins, structure, quartile)
    if flagCompute == True:
        accuracyTrain, accuracyTest, synapses = main(encoding, filterbank, channels, bins, structure, quartile)

        ##### Save data for performance #####
        try:
            performanceData = pd.read_csv(fileName, dtype=str)
            performanceData = performanceData.values.tolist()
            performanceData.append([filterbank, channels, bins, encoding, structure, quartile, synapses, accuracyTrain, accuracyTest])
            performanceData = pd.DataFrame(performanceData, index=None, columns=columnLabels)
            performanceData.to_csv(fileName, index=False)
        except:
            performanceData = [[filterbank, channels, bins, encoding, structure, quartile, synapses, accuracyTrain, accuracyTest]]
            performanceData = pd.DataFrame(performanceData, index=None, columns=columnLabels)
            performanceData.to_csv(fileName, index=False)
