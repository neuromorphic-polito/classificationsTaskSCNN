import warnings
from scipy.io.wavfile import read
import numpy as np
from scipy.signal import lfilter
import copy
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', message='.*Chunk.*')


############################################################
# ##### Dataset standardization and data elaboration ##### #
############################################################
##### Dataset standardization FSD #####
class DataAudio:
    def __init__(self, sampleDir):
        self.name = sampleDir.split('/')[-1].replace('.wav', '')

        self.fs, self.data = read(sampleDir)
        self.freqRange = (20.0, np.floor(self.fs/2))

    def padding(self, width):
        padding = (width+2-self.data.size)//2
        self.data = np.pad(self.data, padding, 'constant')[:width]

    def decomposition(self, filterbank):
        self.freqComp = np.vstack([lfilter(num, den, self.data) for num, den in filterbank])


##### Dataset standardization WISDM #####
class DataDevice:
    def __init__(self, sample, fs):

        self.data = np.vstack(sample['acce'][1:]+sample['gyro'][1:])
        self.fs = fs
        self.freqRange = (0.5, np.floor(self.fs/2))

    def padding(self, width):
        padding = (width+2-self.data.shape[1])//2
        self.data = np.pad(self.data, ((0, 0), (padding, padding)), 'constant')[:, :width]

    def decomposition(self, filterbank):
        self.freqComp = np.vstack([lfilter(num, den, axis) for axis in self.data for num, den in filterbank])


##### Dataset Rate encoding for SCNN #####
class Dataset:
    def __init__(self, dataset, timeStimulus, encoding):

        self.shape = dataset['trainSet'][0][0].shape

        self.trainSet = self._encoding(dataset['trainSet'], timeStimulus, encoding)
        self.testSet = self._encoding(dataset['testSet'], timeStimulus, encoding)

        self.trainSetSpike = self._concatenation(self.trainSet, timeStimulus)
        self.testSetSpike = self._concatenation(self.testSet, timeStimulus)

    def _concatenation(self, dataset, timeStimulus):
        setSpike = [[] for _ in range(self.shape[0]*self.shape[1])]
        for i, sample in enumerate(dataset):
            for pixel, t in enumerate(sample.spike):
                setSpike[pixel].append(t+i*(timeStimulus['duration']+timeStimulus['silence']))
        setSpike = [np.hstack(setSpike[pixel]) for pixel in range(self.shape[0]*self.shape[1])]
        return setSpike

    def _encoding(self, dataset, timeStimulus, encoding):
        numSample = dataset[1].shape[0]
        datasetEncoded = [Encoder(dataset[0][i], dataset[1][1], timeStimulus, encoding) for i in range(numSample)]
        return datasetEncoded


##### Dataset Rate encoding for SCNN #####
class Encoder:
    def __init__(self, source, target, timeStimulus, encoding):

        self.source = source
        self.target = target

        ##### Poisson rate encoding #####
        if encoding == 'poisson':
            np.random.seed(0)

            interval = int(timeStimulus['duration'])

            signal = copy.deepcopy(self.source)
            signal = signal.flatten()
            signal = np.where(signal > 0, signal, 0)

            self.spike = []
            for i, rate in enumerate(signal):
                t = np.array([])
                if rate > 0:
                    ISI = -np.log(1-np.random.random(interval))/rate*interval
                    t = np.cumsum(ISI)
                    t = np.delete(t, np.argwhere(t >= interval))
                self.spike.append(t)

    ##### Plot sample #####
    def plotSample(self):
        plt.title('TBR')
        plt.imshow(self.source, cmap='viridis')
        plt.xlabel('Bins')
        plt.ylabel('Channels')
        plt.show()

    ##### Plot spike train #####
    def plotSpikeTrain(self):
        plt.title('TBR')
        plt.eventplot(self.spike)
        plt.xlabel('Time')
        plt.ylabel('Pixel')
        plt.show()
