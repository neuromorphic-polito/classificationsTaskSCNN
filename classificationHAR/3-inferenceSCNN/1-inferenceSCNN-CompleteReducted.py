import sys
sys.path.append('../../')
import argparse
from utils import datasetSplitting
from utils import Dataset
import tensorflow as tf
from utils import CNN, SNN, Relu
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
def main(datasetName, encoding, filterbank, channels, bins, structure, reduction):
    ###############################
    # ##### Dataset loading ##### #
    ###############################
    sourceFolder = '../../datasets/HumanActivityRecognition/datasetSonograms/'
    fileName = f'{sourceFolder}sonogram_{datasetName}{filterbank}{channels}x{bins}{encoding}.bin'
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
    sourceFolder = '../../networkModels/HumanActivityRecognition/complete/'
    modelCNN = tf.keras.models.load_model(f'{sourceFolder}{datasetName}{filterbank}{channels}x{bins}{structure}{encoding}.keras', custom_objects={'Relu': Relu})

    modelCNN = CNN(modelCNN)

    ###################################
    # ##### Simulation with SNN ##### #
    ###################################
    reductionCode = [int(i) for i in reduction]
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

    modelSNN = SNN(dataset.shape, dataset.trainSetSpike, modelCNN, lifParams, reductionCode)

    spikeTrainNest = modelSNN.start_simulation(len(dataset.trainSet), timeStimulus)
    accuracyTrain = spikeLabeling(numClass, trainTarget, timeSimulation, spikeTrainNest)

    modelSNN = SNN(dataset.shape, dataset.testSetSpike, modelCNN, lifParams, reductionCode)

    spikeTestNest = modelSNN.start_simulation(len(dataset.testSet), timeStimulus)
    accuracyTest = spikeLabeling(numClass, testTarget, timeSimulation, spikeTestNest)

    return accuracyTrain, accuracyTest, modelSNN.synapsesNum


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-n', '--datasetName', help='Dataset file name', type=str, default='subset1')
    parser.add_argument('-e', '--encoding', help='Encoding algorithm selected', type=str, default='RATE')
    parser.add_argument('-f', '--filterbank', help='Type of filterbank', type=str, default='butterworth')
    parser.add_argument('-c', '--channels', help='Frequency decomposition channels', type=int, default=4)
    parser.add_argument('-b', '--bins', help='Binning width', type=int, default=24)
    parser.add_argument('-s', '--structure', help='Network structure', type=str, default='c06c12f2')
    parser.add_argument('-r', '--reduction', help='Synaptic reduction based on the weights', type=str, default='000000')

    argument = parser.parse_args()

    datasetName = argument.datasetName
    encoding = argument.encoding
    filterbank = argument.filterbank
    channels = argument.channels
    bins = argument.bins
    structure = argument.structure
    reduction = argument.reduction

    ##### Check model already calculated #####
    columnLabels = ['Filterbank', 'Channels', 'Bins', 'Encoding', 'Structure', 'Reduction', 'Synapses', 'Train', 'Test']
    flagCompute = True
    sourceFolder = '../../networkPerformance/HumanActivityRecognition/'
    fileName = f'{sourceFolder}{datasetName}SCNN-ModelCompleteReduced.csv'
    try:
        performanceData = pd.read_csv(fileName)
        flagCompute = not bool(len(performanceData[
            (performanceData['Encoding'] == encoding) &
            (performanceData['Filterbank'] == filterbank) &
            (performanceData['Channels'] == channels) &
            (performanceData['Bins'] == bins) &
            (performanceData['Structure'] == structure) &
            (performanceData['Reduction'] == int(reduction))
        ]))
    except:
        pass

    print(encoding, filterbank, channels, bins, structure, reduction)
    if flagCompute == True:
        accuracyTrain, accuracyTest, synapses = main(datasetName, encoding, filterbank, channels, bins, structure, reduction)

        ##### Save data for performance #####
        try:
            performanceData = pd.read_csv(fileName, dtype=str)
            performanceData = performanceData.values.tolist()
            performanceData.append([filterbank, channels, bins, encoding, structure, reduction, synapses, accuracyTrain, accuracyTest])
            performanceData = pd.DataFrame(performanceData, index=None, columns=columnLabels)
            performanceData.to_csv(fileName, index=False)
        except:
            performanceData = [[filterbank, channels, bins, encoding, structure, reduction, synapses, accuracyTrain, accuracyTest]]
            performanceData = pd.DataFrame(performanceData, index=None, columns=columnLabels)
            performanceData.to_csv(fileName, index=False)
