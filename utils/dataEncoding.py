import copy
import numpy as np


class RateCoding:

    def __init__(self, settings):
        self.settings = settings

    def _PoissonRate(self, signal):
        np.random.seed(0)
        interval = self.settings['prFrequencySampling']

        signal = np.where(signal>0, signal, 0)
        signal = np.mean(signal.reshape(-1, interval), axis=1)
        signalMax = signal.max()
        if signalMax > 0:
            signal = signal/signalMax

        spike = np.zeros((signal.shape[0], interval))

        bins = np.linspace(0, 1, interval+1)
        for i, rate in enumerate(signal):
            if rate > 0:
                ISI = -np.log(1-np.random.random(interval))/(rate*interval)
                t = np.searchsorted(bins, np.cumsum(ISI))-1
                t = np.delete(t, np.argwhere(t >= interval))
                spike[i, t] = 1

        spike = spike.reshape(-1)

        return spike

    def PoissonRate(self, signal):
        self.PoissonRateSpike = copy.deepcopy(signal)
        channels = self.PoissonRateSpike.shape[0]
        for channel in range(channels):
            self.PoissonRateSpike[channel] = self._PoissonRate(self.PoissonRateSpike[channel])
        self.PoissonRateSpike = self.PoissonRateSpike.astype(bool)
        self.PoissonRateSpikeAer = np.argwhere(self.PoissonRateSpike == True)


class TemporalContrast:

    def __init__(self, settings):
        self.settings = settings

    def _ThresholdBasedRepresentation(self, signal):
        spike = np.zeros_like(signal)

        factor = self.settings['tbrFactor']

        variation = signal[1:]-signal[:-1]
        threshold = np.mean(variation)+factor*np.std(variation)
        variation = np.insert(variation, 0, variation[1])
        for t, value in enumerate(variation):
            if value > threshold:
                spike[t] = 1
            elif value < -threshold:
                spike[t] = -1

        return spike

    def _StepForward(self, signal):
        spike = np.zeros_like(signal)

        threshold = self.settings['sfThreshold']

        base = signal[0]
        for t, value in enumerate(signal):
            if value > base+threshold:
                spike[t] = 1
                base += threshold
            elif value < base-threshold:
                spike[t] = -1
                base -= threshold

        return spike

    def _ZeroCrossStepForward(self, signal):
        spike = np.zeros_like(signal)

        threshold = self.settings['zcsfThreshold']

        signal = np.where(signal > 0, signal, 0)

        for t, value in enumerate(signal):
            if value > threshold:
                spike[t] = 1

        return spike

    def _MovingWindow(self, c, signal):
        spike = np.zeros_like(signal)

        threshold = self.settings['mwThresholds'][c]
        window = self.settings['mwWindow']

        for t, value in enumerate(signal):
            if t < window:
                base = np.mean(signal[0:window])
            else:
                base = np.mean(signal[t-window:t])
            if value > base+threshold:
                spike[t] = 1
            elif value < base-threshold:
                spike[t] = -1

        return spike

    def ThresholdBasedRepresentation(self, signal):
        self.ThresholdBasedRepresentationSpike = copy.deepcopy(signal)
        channels = self.ThresholdBasedRepresentationSpike.shape[0]
        for channel in range(channels):
            self.ThresholdBasedRepresentationSpike[channel] = self._ThresholdBasedRepresentation(self.ThresholdBasedRepresentationSpike[channel])
        self.ThresholdBasedRepresentationSpike = self.ThresholdBasedRepresentationSpike.astype(bool)
        self.ThresholdBasedRepresentationSpikeAer = np.argwhere(self.ThresholdBasedRepresentationSpike == True)

    def StepForward(self, signal):
        self.StepForwardSpike = copy.deepcopy(signal)
        channels = self.StepForwardSpike.shape[0]
        for channel in range(channels):
            self.StepForwardSpike[channel] = self._StepForward(self.StepForwardSpike[channel])
        self.StepForwardSpike = self.StepForwardSpike.astype(bool)
        self.StepForwardSpikeAer = np.argwhere(self.StepForwardSpike == True)

    def ZeroCrossStepForward(self, signal):
        self.ZeroCrossStepForwardSpike = copy.deepcopy(signal)
        channels = self.ZeroCrossStepForwardSpike.shape[0]
        for channel in range(channels):
            self.ZeroCrossStepForwardSpike[channel] = self._ZeroCrossStepForward(self.ZeroCrossStepForwardSpike[channel])
        self.ZeroCrossStepForwardSpike = self.ZeroCrossStepForwardSpike.astype(bool)
        self.ZeroCrossStepForwardSpikeAer = np.argwhere(self.ZeroCrossStepForwardSpike == True)

    def MovingWindow(self, signal):
        self.MovingWindowSpike = copy.deepcopy(signal)
        channels = self.MovingWindowSpike.shape[0]
        for channel in range(channels):
            self.MovingWindowSpike[channel] = self._MovingWindow(channel, self.MovingWindowSpike[channel])
        self.MovingWindowSpike = self.MovingWindowSpike.astype(bool)
        self.MovingWindowSpikeAer = np.argwhere(self.MovingWindowSpike == True)


class DeconvolutionBased:

    def __init__(self, settings):
        self.settings = settings

    def _HoughSpikerAlgorithm(self, signal):
        spike = np.zeros_like(signal)

        filterWindow = self.settings['hsaFilter']

        filterLength = len(filterWindow)
        signalCopy = copy.deepcopy(signal)

        for t, value in enumerate(signalCopy):
            counter = 0
            for c in range(filterLength):
                if t+c < signalCopy.shape[0] and signalCopy[t+c] >= filterWindow[c]:
                    counter += 1
            if counter == filterLength:
                for c in range(filterLength):
                    if t+c < signalCopy.shape[0]:
                        signalCopy[t+c] -= filterWindow[c]
                spike[t] = 1

        return spike
    def _ModifiedHoughSpikerAlgorithm(self, signal):
        spike = np.zeros_like(signal)

        filterWindow = self.settings['mhsaFilter']

        threshold = self.settings['mhsaThreshold']
        filterLength = len(filterWindow)
        signalCopy = copy.deepcopy(signal)

        for t, value in enumerate(signalCopy):
            error = 0
            for c in range(filterLength):
                if t+c < signalCopy.shape[0] and signalCopy[t+c] < filterWindow[c]:
                    error += (filterWindow[c]-signalCopy[t+c])
            if error <= threshold:
                for c in range(filterLength):
                    if t+c < signalCopy.shape[0]:
                        signalCopy[t+c] -= filterWindow[c]
                spike[t] = 1

        return spike

    def _BenSpikerAlgorithm(self, signal):
        spike = np.zeros_like(signal)

        filterWindow = self.settings['bsaFilter']
        threshold = self.settings['bsaThreshold']

        filterLength = len(filterWindow)
        signalCopy = copy.deepcopy(signal)

        for t, value in enumerate(signalCopy[:-filterLength+1]):
            error1, error2 = 0, 0
            for c in range(filterLength):
                if t+c < signalCopy.shape[0]:
                    error1 += np.abs(signalCopy[t+c]-filterWindow[c])
                    error2 += np.abs(signalCopy[t+c])
            if error1 <= (error2-threshold):
                for c in range(filterLength):
                    if t+c < signalCopy.shape[0]:
                        signalCopy[t+c] -= filterWindow[c]
                spike[t] = 1

        return spike

    def HoughSpikerAlgorithm(self, signal):
        self.HoughSpikerAlgorithmSpike = copy.deepcopy(signal)
        channels = self.HoughSpikerAlgorithmSpike.shape[0]
        for channel in range(channels):
            self.HoughSpikerAlgorithmSpike[channel] = self._HoughSpikerAlgorithm(self.HoughSpikerAlgorithmSpike[channel])
        self.HoughSpikerAlgorithmSpike = self.HoughSpikerAlgorithmSpike.astype(bool)
        self.HoughSpikerAlgorithmSpikeAer = np.argwhere(self.HoughSpikerAlgorithmSpike == True)

    def ModifiedHoughSpikerAlgorithm(self, signal):
        self.ModifiedHoughSpikerAlgorithmSpike = copy.deepcopy(signal)
        channels = self.ModifiedHoughSpikerAlgorithmSpike.shape[0]
        for channel in range(channels):
            self.ModifiedHoughSpikerAlgorithmSpike[channel] = self._ModifiedHoughSpikerAlgorithm(self.ModifiedHoughSpikerAlgorithmSpike[channel])
        self.ModifiedHoughSpikerAlgorithmSpike = self.ModifiedHoughSpikerAlgorithmSpike.astype(bool)
        self.ModifiedHoughSpikerAlgorithmSpikeAer = np.argwhere(self.ModifiedHoughSpikerAlgorithmSpike == True)

    def BenSpikerAlgorithm(self, signal):
        self.BenSpikerAlgorithmSpike = copy.deepcopy(signal)
        channels = self.BenSpikerAlgorithmSpike.shape[0]
        for channel in range(channels):
            self.BenSpikerAlgorithmSpike[channel] = self._BenSpikerAlgorithm(self.BenSpikerAlgorithmSpike[channel])
        self.BenSpikerAlgorithmSpike = self.BenSpikerAlgorithmSpike.astype(bool)
        self.BenSpikerAlgorithmSpikeAer = np.argwhere(self.BenSpikerAlgorithmSpike == True)


class GlobalReferenced:
    def __init__(self, settings):
        self.settings = settings

    def _PhaseEncoding(self, signal):
        bit = self.settings['peBit']

        signal = np.where(signal > 0, signal, 0)
        signal = np.mean(signal.reshape(-1, bit), axis=1)
        signalMax = signal.max()
        if signalMax > 0:
            signal = signal/signalMax

        phase = np.arcsin(signal)
        bins = np.linspace(0, np.pi/2, 2 ** bit+1)
        levels = np.searchsorted(bins, phase)
        levels = np.where(levels == 2 ** bit, levels-1, levels)
        spike = np.array([list(map(int, list(f'{level:0{bit}b}'))) for level in levels]).reshape(-1)

        return spike

    def _TimeToFirstSpike(self, signal):
        interval = self.settings['ttfsInterval']

        signal = np.where(signal > 0, signal, 0)
        signal = np.mean(signal.reshape(-1, interval), axis=1)
        signalMax = signal.max()
        if signalMax > 0:
            signal = signal/signalMax

        intensity = np.ones_like(signal)*2
        for i, value in enumerate(signal):
            if value > 0:
                intensity[i] = 0.1*np.log(1/value)
        bins = np.linspace(0, 1, interval)
        levels = np.searchsorted(bins, intensity)

        spike = np.zeros((signal.shape[0], interval+1))
        for i, level in enumerate(levels):
            spike[i, level] = 1
        spike = spike[:, :-1].reshape(-1)

        return spike

    def PhaseEncoding(self, signal):
        _, timeStamp = signal.shape
        self.PhaseEncodingSpike = copy.deepcopy(signal)
        bit = self.settings['peBit']
        if timeStamp % bit > 0:
            self.PhaseEncodingSpike = self.PhaseEncodingSpike[:, 0:timeStamp//bit*bit]
        channels = self.PhaseEncodingSpike.shape[0]
        for channel in range(channels):
            self.PhaseEncodingSpike[channel] = self._PhaseEncoding(self.PhaseEncodingSpike[channel])
        self.PhaseEncodingSpike = np.hstack([self.PhaseEncodingSpike, np.zeros((channels, timeStamp-timeStamp//bit*bit))])
        self.PhaseEncodingSpike = self.PhaseEncodingSpike.astype(bool)
        self.PhaseEncodingSpikeAer = np.argwhere(self.PhaseEncodingSpike == True)


    def TimeToFirstSpike(self, signal):
        self.TimeToFirstSpikeSpike = copy.deepcopy(signal)
        channels = self.TimeToFirstSpikeSpike.shape[0]
        for channel in range(channels):
            self.TimeToFirstSpikeSpike[channel] = self._TimeToFirstSpike(self.TimeToFirstSpikeSpike[channel])
        self.TimeToFirstSpikeSpike = self.TimeToFirstSpikeSpike.astype(bool)
        self.TimeToFirstSpikeSpikeAer = np.argwhere(self.TimeToFirstSpikeSpike == True)


class Latency:

    def __init__(self, settings):
        self.settings = settings

    def _BurstEncoding(self, signal):
        nMax = self.settings['beNmax']
        tMin = self.settings['beTmin']
        tMax = self.settings['beTmax']
        length = self.settings['beLength']

        signal = np.where(signal > 0, signal, 0)
        signal = np.mean(signal.reshape(-1, length), axis=1)
        signalMax = signal.max()
        if signalMax > 0:
            signal = signal/signalMax

        spikeNum = np.ceil(signal*nMax).astype(int)
        ISI = np.ceil(tMax-signal*(tMax-tMin)).astype(int)

        if length < np.max(spikeNum*(ISI+1)):
            raise ValueError(f'Invalid stream length, the min length is {np.max(spikeNum*(ISI+1))}')

        spike = []
        for i in range(len(signal)):
            code = ([1]+ISI[i]*[0])*spikeNum[i]
            code += [0]*(length-len(code))
            spike.append(code)

        spike = np.array(spike).reshape(-1)

        return spike

    def BurstEncoding(self, signal):
        _, timeStamp = signal.shape
        self.BurstEncodingSpike = copy.deepcopy(signal)
        length = self.settings['beLength']
        if timeStamp % length > 0:
            self.BurstEncodingSpike = self.BurstEncodingSpike[:, 0:timeStamp//length*length]
        channels = self.BurstEncodingSpike.shape[0]
        for channel in range(channels):
            self.BurstEncodingSpike[channel] = self._BurstEncoding(self.BurstEncodingSpike[channel])
        self.BurstEncodingSpike = np.hstack([self.BurstEncodingSpike, np.zeros((channels, timeStamp-timeStamp//length*length))])
        self.BurstEncodingSpike = self.BurstEncodingSpike.astype(bool)
        self.BurstEncodingSpikeAer = np.argwhere(self.BurstEncodingSpike == True)
