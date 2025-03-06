import sys
sys.path.append('../../')
import pickle
from utils import DataDevice
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, gammatone, freqz

######################################
# ##### Sample standardization ##### #
######################################
channels, pad = 4, 4800
file = open('../../datasets/HumanActivityRecognition/datasetRaw/datasetsWisdm.bin', 'rb')
dataset = pickle.load(file)
file.close()
sample = DataDevice(dataset['watch'][0]['A'], 20)
sample.padding(pad)


###########################
# ##### Sample plot ##### #
###########################
plt.figure()
plt.title('WISDM Sample')
for axis in range(6):
    plt.subplot(2, 3, axis+1)
    plt.plot(np.linspace(0, 180, pad), sample.data[axis])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
plt.tight_layout()


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


####################################################
# ##### Frequency decomposition and Plotting ##### #
####################################################
##### Spectrogram #####
sample.decomposition(filterbankButter)
spectrogram = np.absolute(sample.freqComp)
plt.figure()
plt.title('Spectrogram with Butterworth filterbank')
plt.imshow(spectrogram, aspect='auto', vmax=spectrogram.max()/10)
plt.xlabel('Time')
plt.ylabel('Channels')

##### Frequency components #####
plt.figure()
for i in range(6*channels):
    plt.subplot(24, 1, i+1)
    plt.plot(sample.freqComp[i]/sample.freqComp[i].max())
    plt.axis('off')
plt.xlabel('Sample')
plt.tight_layout()

plt.show()
