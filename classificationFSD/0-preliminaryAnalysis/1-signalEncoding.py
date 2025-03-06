import sys
sys.path.append('../../')
from utils import DataAudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, gammatone, freqz
from utils import RateCoding, TemporalContrast, DeconvolutionBased, GlobalReferenced, Latency
from utils import spikeEfficiency
from scipy.signal.windows import *

######################################
# ##### Sample standardization ##### #
######################################
channels, pad = 16, 8000
sourceFolder = '../../datasets/FreeSpokenDigits/datasetRaw/'
sample = DataAudio(f'{sourceFolder}0_jackson_0.wav')
sample.padding(pad)


###########################
# ##### Filter bank ##### #
###########################
##### Filter bank parameter #####
freqMin, freqMax = sample.freqRange
octave = (channels-0.5)*np.log10(2)/np.log10(freqMax/freqMin)
freqCentr = np.array([freqMin*(2**(ch/octave)) for ch in range(channels)])
freqPoles = np.array([(freq*(2**(-1/(2*octave))), (freq*(2**(1/(2*octave))))) for freq in freqCentr])
freqPoles[-1, 1] = sample.fs/2*0.99999

##### Butterworth filter banks #####
order = 2
plt.figure()
plt.title('Butterworth filter banks')
filterbankButter = []
for freqLow, freqHigh in freqPoles:
    num, den = butter(N=order, Wn=(freqLow, freqHigh), btype='band', fs=sample.fs)
    filterbankButter.append([num, den])
    f, h = freqz(num, den, worN=20000)
    plt.plot((sample.fs*0.5/np.pi)*f, abs(h))
plt.xlabel('f(Hz)')
plt.ylabel('Gain')
sample.decomposition(filterbankButter)
"""
##### Gammatone filterbank #####
order = 1
plt.figure()
plt.title('Gammatone filter banks')
filterbankGammatone = []
for freq in freqCentr:
    num, den = gammatone(order=order, freq=freq, ftype='fir', fs=sample.fs)
    filterbankGammatone.append([num, den])
    f, h = freqz(num, den, worN=20000)
    plt.plot((sample.fs*0.5/np.pi)*f, abs(h))
plt.xlabel('f(Hz)')
plt.ylabel('Gain')
sample.decomposition(filterbankGammatone)
"""

########################
# ##### Encoding ##### #
########################
labelEncoding = []
spikeTrain = []

##### Rate coding #####
settings = {'prFrequencySampling': 20}
encoder = RateCoding(settings)
encoder.PoissonRate(sample.freqComp)
labelEncoding.append('RATE')
spikeTrain.append(encoder.PoissonRateSpike)


##### Temporal contrast #####
sfThreshold = np.mean([component.max()-component.min() for component in sample.freqComp])/10
zcsfThreshold = np.mean([component.max()-component.min() for component in sample.freqComp])/10
mwThreshold = [np.mean(np.abs(component[1:]-component[:-1])) for component in sample.freqComp]

settings = {
    'tbrFactor': 1,
    'sfThreshold': sfThreshold,
    'zcsfThreshold': zcsfThreshold,
    'mwWindow': 3, 'mwThresholds': mwThreshold,
}

encoder = TemporalContrast(settings)

encoder.ThresholdBasedRepresentation(sample.freqComp)
labelEncoding.append('TBR')
spikeTrain.append(encoder.ThresholdBasedRepresentationSpike)

encoder.StepForward(sample.freqComp)
labelEncoding.append('SF')
spikeTrain.append(encoder.StepForwardSpike)

encoder.ZeroCrossStepForward(sample.freqComp)
labelEncoding.append('ZCSF')
spikeTrain.append(encoder.ZeroCrossStepForwardSpike)

encoder.MovingWindow(sample.freqComp)
labelEncoding.append('MW')
spikeTrain.append(encoder.MovingWindowSpike)


##### Deconvolution based #####
filterWindow = boxcar(3)
settings = {
    'hsaFilter': filterWindow,
    'mhsaFilter': filterWindow, 'mhsaThreshold': 0.85,
    'bsaFilter': filterWindow, 'bsaThreshold': 1,
}

encoder = DeconvolutionBased(settings)

encoder.HoughSpikerAlgorithm(sample.freqComp)
labelEncoding.append('HSA')
spikeTrain.append(encoder.HoughSpikerAlgorithmSpike)

encoder.ModifiedHoughSpikerAlgorithm(sample.freqComp)
labelEncoding.append('MHSA')
spikeTrain.append(encoder.ModifiedHoughSpikerAlgorithmSpike)

encoder.BenSpikerAlgorithm(sample.freqComp)
labelEncoding.append('BSA')
spikeTrain.append(encoder.BenSpikerAlgorithmSpike)


##### Global Referenced #####
settings = {
    'peBit': 6,
    'ttfsInterval': 10,
}

encoder = GlobalReferenced(settings)

encoder.PhaseEncoding(sample.freqComp)
labelEncoding.append('PHASE')
spikeTrain.append(encoder.PhaseEncodingSpike)

encoder.TimeToFirstSpike(sample.freqComp)
labelEncoding.append('TTFS')
spikeTrain.append(encoder.TimeToFirstSpikeSpike)


##### Latency #####
settings = {'beNmax': 5, 'beTmin': 0, 'beTmax': 4, 'beLength': 13}

encoder = Latency(settings)

encoder.BurstEncoding(sample.freqComp)

labelEncoding.append('BURST')
spikeTrain.append(encoder.BurstEncodingSpike)


########################
# ##### Plotting ##### #
########################
##### Encoding example #####
indexEncoding = len(labelEncoding)
spike = [np.flip(i, 0) for i in spikeTrain]
for index in range(indexEncoding):
    plt.figure()
    plt.title(labelEncoding[index])
    plt.eventplot(np.absolute(spike[index])*np.linspace(0, 1, pad), linelengths=0.9)
    plt.yticks([])
    plt.ylabel('Channel')
    plt.xlabel('Time')

##### Spike efficiency #####
plt.figure()
plt.title('Efficiency')
for index in range(indexEncoding):
    efficiency = spikeEfficiency(spike[index])
    plt.plot(efficiency, '-o')
plt.xlabel('Channel')
plt.ylabel('Efficiency')
plt.legend(labelEncoding)

plt.show()
