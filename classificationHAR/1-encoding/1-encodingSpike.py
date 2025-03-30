import sys
sys.path.append('../../')
import argparse
import numpy as np
from utils import DataDevice
from scipy.signal import butter, gammatone
from scipy.signal.windows import *
from utils import RateCoding, TemporalContrast, DeconvolutionBased, GlobalReferenced, Latency
import pickle


def main(device, datasetName, subsetLabel, filterbank, channels):
    freqMin, freqMax = (0.5, 10)
    octave = (channels-0.5)*np.log10(2)/np.log10(freqMax/freqMin)
    freqCentr = np.array([freqMin*(2**(ch/octave)) for ch in range(channels)])
    freqPoles = np.array([(freq*(2**(-1/(2*octave))), (freq*(2**(1/(2*octave))))) for freq in freqCentr])
    freqPoles[-1, 1] = 20/2*0.99999

    encodings = ['RATE', 'TBR', 'SF', 'ZCSF', 'MW', 'HSA', 'MHSA', 'BSA', 'PHASE', 'TTFS', 'BURST']

    file = open('../../datasets/HumanActivityRecognition/datasetRaw/datasetsWisdm.bin', 'rb')
    dataset = pickle.load(file)
    file.close()
    dataset = dataset[device]

    #############################
    # ##### Data encoding ##### #
    #############################
    datasetSpike = [[] for _ in range(len(encodings))]
    for subject in range(51):
        labels = dataset[subject].keys()
        for idxlabel, label in enumerate(subsetLabel):
            if label in labels:
                # ##### Sample standardization ##### #
                # print(f'Sample: {subject+1}/51, Label {label}')
                sample = DataDevice(dataset[subject][label], 20)
                sample.padding(4800)

                # ##### Frequency decomposition ##### #
                if filterbank == 'butterworth':
                    order = 2
                    filterbankButter = []
                    for freqLow, freqHigh in freqPoles:
                        num, den = butter(N=order, Wn=(freqLow, freqHigh), btype='band', fs=sample.fs)
                        filterbankButter.append([num, den])
                    sample.decomposition(filterbankButter)
                elif filterbank == 'gammatone':
                    order = 1
                    filterbankGammatone = []
                    for freq in freqCentr:
                        num, den = gammatone(order=order, freq=freq, ftype='fir', fs=sample.fs)
                        filterbankGammatone.append([num, den])
                    sample.decomposition(filterbankGammatone)


                # ##### Parameters definition ##### #
                sfThreshold = np.mean([component.max()-component.min() for component in sample.freqComp])/10
                zcsfThreshold = np.mean([component.max()-component.min() for component in sample.freqComp])/10
                mwThreshold = [np.mean(np.abs(component[1:]-component[:-1])) for component in sample.freqComp]
                filterWindow = boxcar(3)
                settings = {
                    'prFrequencySampling': 20,
                    'tbrFactor': 0.5,
                    'sfThreshold': sfThreshold,
                    'zcsfThreshold': zcsfThreshold,
                    'mwWindow': 3, 'mwThresholds': mwThreshold,
                    'hsaFilter': filterWindow,
                    'mhsaFilter': filterWindow, 'mhsaThreshold': 0.85,
                    'bsaFilter': filterWindow, 'bsaThreshold': 1,
                    'peBit': 5,
                    'ttfsInterval': 10,
                    'beNmax': 5, 'beTmin': 0, 'beTmax': 4, 'beLength': 14
                }


                # ##### Encoding settings ##### #
                ##### Rate coding #####
                encoderRateCoding = RateCoding(settings)
                encoderRateCoding.PoissonRate(sample.freqComp)

                ##### Temporal contrast #####
                encoderTemporalContrast = TemporalContrast(settings)
                encoderTemporalContrast.ThresholdBasedRepresentation(sample.freqComp)
                encoderTemporalContrast.StepForward(sample.freqComp)
                encoderTemporalContrast.ZeroCrossStepForward(sample.freqComp)
                encoderTemporalContrast.MovingWindow(sample.freqComp)

                ##### Deconvolution based #####
                encoderDeconvolutionBased = DeconvolutionBased(settings)
                encoderDeconvolutionBased.HoughSpikerAlgorithm(sample.freqComp)
                encoderDeconvolutionBased.ModifiedHoughSpikerAlgorithm(sample.freqComp)
                encoderDeconvolutionBased.BenSpikerAlgorithm(sample.freqComp)

                ##### Global referenced #####
                encoderGlobalReferenced = GlobalReferenced(settings)
                encoderGlobalReferenced.PhaseEncoding(sample.freqComp)
                encoderGlobalReferenced.TimeToFirstSpike(sample.freqComp)

                ##### Latency #####
                encoderLatency = Latency(settings)
                encoderLatency.BurstEncoding(sample.freqComp)


                ###############################
                # ##### Data formatting ##### #
                ###############################
                ##### Rate coding #####
                datasetSpike[0].append([encoderRateCoding.PoissonRateSpikeAer[:, 0], encoderRateCoding.PoissonRateSpikeAer[:, 1], idxlabel])

                ##### Temporal contrast #####
                datasetSpike[1].append([encoderTemporalContrast.ThresholdBasedRepresentationSpikeAer[:, 0], encoderTemporalContrast.ThresholdBasedRepresentationSpikeAer[:, 1], idxlabel])
                datasetSpike[2].append([encoderTemporalContrast.StepForwardSpikeAer[:, 0], encoderTemporalContrast.StepForwardSpikeAer[:, 1], idxlabel])
                datasetSpike[3].append([encoderTemporalContrast.ZeroCrossStepForwardSpikeAer[:, 0], encoderTemporalContrast.ZeroCrossStepForwardSpikeAer[:, 1], idxlabel])
                datasetSpike[4].append([encoderTemporalContrast.MovingWindowSpikeAer[:, 0], encoderTemporalContrast.MovingWindowSpikeAer[:, 1], idxlabel])

                ##### Deconvolution based #####
                datasetSpike[5].append([encoderDeconvolutionBased.HoughSpikerAlgorithmSpikeAer[:, 0], encoderDeconvolutionBased.HoughSpikerAlgorithmSpikeAer[:, 1], idxlabel])
                datasetSpike[6].append([encoderDeconvolutionBased.ModifiedHoughSpikerAlgorithmSpikeAer[:, 0], encoderDeconvolutionBased.ModifiedHoughSpikerAlgorithmSpikeAer[:, 1], idxlabel])
                datasetSpike[7].append([encoderDeconvolutionBased.BenSpikerAlgorithmSpikeAer[:, 0], encoderDeconvolutionBased.BenSpikerAlgorithmSpikeAer[:, 1], idxlabel])

                ##### Global referenced #####
                datasetSpike[8].append([encoderGlobalReferenced.PhaseEncodingSpikeAer[:, 0], encoderGlobalReferenced.PhaseEncodingSpikeAer[:, 1], idxlabel])
                datasetSpike[9].append([encoderGlobalReferenced.TimeToFirstSpikeSpikeAer[:, 0], encoderGlobalReferenced.TimeToFirstSpikeSpikeAer[:, 1], idxlabel])

                ##### Latency #####
                datasetSpike[10].append([encoderLatency.BurstEncodingSpikeAer[:, 0], encoderLatency.BurstEncodingSpikeAer[:, 1], idxlabel])


    #########################
    # ##### Save data ##### #
    #########################
    sourceFolder = f'../../datasets/HumanActivityRecognition/datasetSpike/'
    for i, encoding in enumerate(encodings):
        file = open(f'{sourceFolder}spikeTrains_{datasetName}{filterbank}{channels}{encoding}.bin', 'wb')
        pickle.dump(datasetSpike[i], file)
        file.close()

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-d', '--device', help='Device dataset', type=str, default='watch')
    parser.add_argument('-n', '--datasetName', help='Dataset file name', type=str, default='subset1')
    parser.add_argument('-s', '--subsetLabel', help='List of subset class', type=str, default='A,B,G,H,P,R')
    parser.add_argument('-f', '--filterbank', help='Type of filterbank', type=str, default='butterworth')
    parser.add_argument('-c', '--channels', help='Frequency decomposition channels', type=int, default=4)

    argument = parser.parse_args()

    device = argument.device
    datasetName = argument.datasetName
    subsetLabel = argument.subsetLabel.split(',')
    filterbank = argument.filterbank
    channels = argument.channels

    main(device, datasetName, subsetLabel, filterbank, channels)
