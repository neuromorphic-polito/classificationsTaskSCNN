import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np


def featureExtraction(datasetSpike, channels, pad, binsWindow, plot=False):
    datasetSonogram = []
    binsSample, binsNumber = int(pad*binsWindow//1000), int(1000//binsWindow)
    for address, t, label in datasetSpike:

        spikeTrain = np.zeros((channels, pad), dtype=bool)
        spikeTrain[address, t] = True

        sonogram = np.zeros((channels, binsNumber))
        for slices in range(binsNumber):
            sonogram[:, slices] = np.sum(spikeTrain[:, 0+slices*binsSample:binsSample+slices*binsSample], axis=1)
        if plot:
            plt.figure()
            plt.imshow(spikeTrain, aspect='auto')
            plt.figure()
            plt.xlabel('Bins')
            plt.ylabel('Channels')
            plt.imshow(sonogram, aspect='auto',  vmin=0, vmax=75)
            plt.show()
        datasetSonogram.append([sonogram, label])
    return binsNumber, datasetSonogram


def main(encoding, filterbank, channels, binsWindow):
    ##################################
    # ##### Load dataset spike ##### #
    ##################################
    sourceFolder = '../../datasets/FreeSpokenDigits/datasetSpike/'
    file = open(f'{sourceFolder}spikeTrains_{filterbank}{channels}{encoding}.bin', 'rb')
    datasetSpike = pickle.load(file)
    file.close()

    #################################
    # ##### Sonogram creation ##### #
    #################################
    bins, sonograms = featureExtraction(datasetSpike, channels, 8000, binsWindow)
    datasetSonograms = {}
    for value, key in sonograms:
        if key not in datasetSonograms.keys():
            datasetSonograms[key] = [value]
        else:
            datasetSonograms[key] += [value]
    classMin = np.min([len(samples) for samples in datasetSonograms.values()])
    datasetSonograms = {key: datasetSonograms[key][0:classMin] for key in datasetSonograms.keys()}

    #####################################
    # ##### Save sonogram dataset ##### #
    #####################################
    sourceFolder = f'../../datasets/FreeSpokenDigits/datasetSonograms/'
    file = open(f'{sourceFolder}sonograms_{filterbank}{channels}x{bins}{encoding}.bin', 'wb')
    pickle.dump(datasetSonograms, file)
    file.close()

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-e', '--encoding', help='Encoding algorithm selected', type=str, default='RATE')
    parser.add_argument('-f', '--filterbank', help='Type of filterbank', type=str, default='butterworth')
    parser.add_argument('-c', '--channels', help='Frequency decomposition channels', type=int, default=32)
    parser.add_argument('-b', '--binsWindow', help='Binning width', type=float, default=20.0)

    argument = parser.parse_args()

    ##### Parsing unpack #####
    encoding = argument.encoding
    filterbank = argument.filterbank
    channels = argument.channels
    binsWindow = argument.binsWindow

    main(encoding, filterbank, channels, binsWindow)
