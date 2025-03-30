import sys
sys.path.append('../../')
import argparse
import pandas as pd
from utils import datasetSplit
from utils import Dataset
import torch
from utils import SNN
import numpy as np
import os


##############################
# ##### Inference loop ##### #
##############################
def main(encoding, filterbank, channels, bins, structure, reduction):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ##### Dataset load ##### #
    sourceFolder = '../../datasets/FreeSpokenDigits/datasetSonograms/'
    fileName = f'{sourceFolder}sonograms_{filterbank}{channels}x{bins}{encoding}.bin'
    trainSource, trainTarget, testSource, testTarget, numClass = datasetSplit(fileName, 'SNN')

    timeStimulus = {'duration': 1000.0, 'silence': 30.0}

    dataset = Dataset(
        {'trainSet': (trainSource, trainTarget), 'testSet': (testSource, testTarget)},
        timeStimulus, 'poisson'
    )

    ##### Load model #####
    sourceFolder = '../../networkModels/FreeSpokenDigits/complete/'
    modelCNN = torch.load(f'{sourceFolder}{filterbank}{channels}x{bins}{structure}{encoding}.pth', weights_only=False)

    # ##### SNN inference ##### #
    ##### Neuron parameter #####
    lifParam = {
        'C': 0.25,  # nF
        'TauM': 20.0,  # ms
        'Ioffset': 0.1,  # nA
        'Vrest': -65.0,  # mV
        'Vthresh': -50.0,  # mV
        'Vreset': -65.0,  # mV
        'TauRefrac': 1.0,  # ms
    }
    lifVar = {
        'V': lifParam['Vrest'],  # mV
        'RefracTime': 0.0,  # ms
    }

    ##### Training set inference #####
    modelSNN = SNN(device, (lifParam, lifVar), 'trainSetSpike', dataset, modelCNN, reduction)
    timeSteps = int((timeStimulus['duration']+timeStimulus['silence'])*trainTarget.size)
    modelSNN.model.build()
    modelSNN.model.load(num_recording_timesteps=timeSteps)
    while modelSNN.model.timestep < timeSteps:
        modelSNN.model.step_time()
    modelSNN.model.pull_recording_buffers_from_device()
    times, index = modelSNN.layerOutput.spike_recording_data[0]
    binsPop = np.arange(0, numClass+1, 1)
    binsTime = np.arange(0, (timeStimulus['duration']+timeStimulus['silence'])*(trainTarget.size+1), (timeStimulus['duration']+timeStimulus['silence']))
    wta = np.histogram2d(index, times, (binsPop, binsTime))[0]
    targetTrue = trainTarget
    targetPred = np.argmax(wta, axis=0)
    accuracyTrain = np.sum(targetPred == targetTrue)/ targetTrue.shape[0]
    os.system('rm -r .*_CODE*')

    ##### Test set inference #####
    modelSNN = SNN(device, (lifParam, lifVar), 'testSetSpike', dataset, modelCNN, reduction)
    timeSteps = int((timeStimulus['duration']+timeStimulus['silence'])*testTarget.size)
    modelSNN.model.build()
    modelSNN.model.load(num_recording_timesteps=timeSteps)
    while modelSNN.model.timestep < timeSteps:
        modelSNN.model.step_time()
    modelSNN.model.pull_recording_buffers_from_device()
    times, index = modelSNN.layerOutput.spike_recording_data[0]
    binsPop = np.arange(0, numClass+1, 1)
    binsTime = np.arange(0, (timeStimulus['duration']+timeStimulus['silence'])*(testTarget.size+1), (timeStimulus['duration']+timeStimulus['silence']))
    wta = np.histogram2d(index, times, (binsPop, binsTime))[0]
    targetTrue = testTarget
    targetPred = np.argmax(wta, axis=0)
    accuracyTest = np.sum(targetPred == targetTrue)/ targetTrue.shape[0]
    os.system('rm -r .*_CODE*')

    return accuracyTrain, accuracyTest, modelSNN.synapsesNum


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-e', '--encoding', help='Encoding algorithm selected', type=str, default='RATE')
    parser.add_argument('-f', '--filterbank', help='Type of filterbank', type=str, default='butterworth')
    parser.add_argument('-c', '--channels', help='Frequency decomposition channels', type=int, default=32)
    parser.add_argument('-b', '--bins', help='Binning width', type=int, default=50)
    parser.add_argument('-s', '--structure', help='Network structure', type=str, default='c06c12f2')
    parser.add_argument('-r', '--reduction', help='Synaptic reduction based on the weights', type=int, default=0)

    argument = parser.parse_args()

    encoding = argument.encoding
    filterbank = argument.filterbank
    channels = argument.channels
    bins = argument.bins
    structure = argument.structure
    reduction = argument.reduction

    ##### Verify stored model #####
    columnLabels = ['Filterbank', 'Channels', 'Bins', 'Encoding', 'Structure', 'Reduction', 'Synapses', 'Train', 'Test']
    flagCompute = True
    sourceFolder = '../../networkPerformance/FreeSpokenDigits/'
    fileName = f'{sourceFolder}SCNN-ModelCompleteReduced.csv'
    try:
        performanceData = pd.read_csv(fileName)
        flagCompute = not bool(len(performanceData[
            (performanceData['Encoding'] == encoding) &
            (performanceData['Filterbank'] == filterbank) &
            (performanceData['Channels'] == channels) &
            (performanceData['Bins'] == bins) &
            (performanceData['Structure'] == structure) &
            (performanceData['Reduction'] == reduction)
        ]))
    except:
        pass

    ##### Training models #####
    # print(encoding, filterbank, channels, bins, structure, reduction)
    if flagCompute == True:
        accuracyTrain, accuracyTest, synapses = main(encoding, filterbank, channels, bins, structure, reduction)

        ##### Save performance #####
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
