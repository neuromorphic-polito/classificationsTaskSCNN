import sys
sys.path.append('../../')
from utils import DataAudio
import numpy as np
from scipy.signal import butter, gammatone, freqz
import matplotlib.pyplot as plt


######################################
# ##### Sample standardization ##### #
######################################
channels, pad = 16, 8000
sourceDir = '../../datasets/FreeSpokenDigits/datasetRaw/'
sample = DataAudio(f'{sourceDir}0_jackson_0.wav')
sample.padding(pad)


###########################
# ##### Sample plot ##### #
###########################
plt.figure()
plt.title('FSD Sample')
plt.plot(np.linspace(0, 1, pad), sample.data)
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
for i in range(channels):
    plt.subplot(16, 1, i+1)
    plt.plot(sample.freqComp[i]/sample.freqComp[i].max())
    plt.axis('off')
plt.xlabel('Sample')
plt.tight_layout()

plt.show()
