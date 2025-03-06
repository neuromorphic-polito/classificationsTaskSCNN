import pyNN.nest as pynn
import numpy as np


class SNN:
    def _kernelMask(self, weights, distribution, code):
        threshold = np.multiply(distribution, code).max()
        musk = np.where(np.abs(weights)>threshold, 1, 0)
        return musk

    def __init__(self, shape, spike, modelCNN, lifParameter, filterCode=(0, 0, 0, 0, 0, 0)):
        self.spike = spike

        self.layers, self.synapsesNum = [], 0
        layersShape = []

        ##### PyNN initialization #####
        pynn.setup(timestep=1.0, threads=7)

        ##### Input layer #####
        dataRow, dataCol = shape
        neuronNum = dataRow*dataCol
        layerInput = pynn.Population(neuronNum, pynn.SpikeSourceArray, {'spike_times': spike})
        self.layers.append([layerInput])
        layersShape.append({'shape': shape, 'neuronNum': neuronNum})

        idxLayer = 0  # previous index layer
        for layer in modelCNN.layers:
            print(layer.type)

            if layer.type == 'conv2d':
                inputRow, inputCol = layersShape[idxLayer]['shape']
                kernelRow, kernelCol = layer.sizeKernel
                outputRow, outputCol = inputRow-kernelRow+1, inputCol-kernelCol+1
                neuronNum = outputRow*outputCol

                self.layers.append([])
                for idxOutput in range(layer.featureOutput):
                    layerConv = pynn.Population(neuronNum, pynn.IF_curr_exp, lifParameter)

                    for idxInput in range(len(self.layers[idxLayer])):
                        weights = layer.weights[idxOutput][idxInput]
                        weightsMask = self._kernelMask(weights, layer.distribution, filterCode)
                        weights *= weightsMask

                        synapseExcit, synapseInhib = [], []
                        for oRow in range(outputRow):
                            for oCol in range(outputCol):
                                idxTarget = oCol+oRow*outputCol
                                for kRow in range(kernelRow):
                                    for kCol in range(kernelCol):
                                        weight = weights[kRow, kCol]
                                        idxSource = oCol+kCol+(oRow+kRow)*inputCol
                                        if weight > 0:
                                            synapseExcit.append((idxSource, idxTarget, weight, 1.0))
                                        elif weight < 0:
                                            synapseInhib.append((idxSource, idxTarget, weight, 1.0))
                        pynn.Projection(self.layers[idxLayer][idxInput], layerConv, pynn.FromListConnector(synapseExcit), receptor_type='excitatory')
                        pynn.Projection(self.layers[idxLayer][idxInput], layerConv, pynn.FromListConnector(synapseInhib), receptor_type='inhibitory')
                        self.synapsesNum += len(synapseExcit)+len(synapseInhib)

                    self.layers[-1].append(layerConv)
                layersShape.append({'shape': (outputRow, outputCol), 'neuronNum': neuronNum})

            elif layer.type == 'pool2d':
                inputRow, inputCol = layersShape[idxLayer]['shape']
                poolRow, poolCol = layer.sizePool
                outputRow, outputCol = int(inputRow/poolRow), int(inputCol/poolCol)
                neuronNum = outputRow*outputCol

                self.layers.append([])
                for idxInput in range(len(self.layers[idxLayer])):
                    layerPool = pynn.Population(neuronNum, pynn.IF_curr_exp, lifParameter)
                    weight = layer.weights[0, 0]

                    synapseExcit = []
                    for oRow in range(outputRow):
                        for oCol in range(outputCol):
                            idxTarget = oCol+oRow*outputCol
                            for pRow in range(poolRow):
                                for pCol in range(poolCol):
                                    idxSource = oCol*poolCol+pCol+(oRow*poolRow+pRow)*inputCol
                                    synapseExcit.append((idxSource, idxTarget, weight, 1.0))

                    pynn.Projection(self.layers[idxLayer][idxInput], layerPool, pynn.FromListConnector(synapseExcit), receptor_type='excitatory')
                    self.synapsesNum += len(synapseExcit)

                    self.layers[-1].append(layerPool)
                layersShape.append({'shape': (outputRow, outputCol), 'neuronNum': neuronNum})
            elif layer.type == 'dense':
                inputRow, inputCol = layersShape[idxLayer]['shape']
                neuronNumInp = inputRow*inputCol

                neuronNumOut = layer.output
                layerDense = pynn.Population(neuronNumOut, pynn.IF_curr_exp, lifParameter)

                if neuronNumInp != 0:
                    weights = layer.weights

                    for idxInput in range(len(self.layers[idxLayer])):
                        weightsSection = weights[:, idxInput*neuronNumInp:(idxInput+1)*neuronNumInp]
                        weightsMask = self._kernelMask(weightsSection, layer.distribution, filterCode)
                        weightsSection *= weightsMask

                        synapseExcit, synapseInhib = [], []
                        for io in range(neuronNumOut):
                            for ii in range(neuronNumInp):
                                weight = weightsSection[io, ii]
                                if weight > 0:
                                    synapseExcit.append((ii, io, weight, 1.0))
                                elif weight < 0:
                                    synapseInhib.append((ii, io, weight, 1.0))

                        pynn.Projection(self.layers[idxLayer][idxInput], layerDense, pynn.FromListConnector(synapseExcit), receptor_type='excitatory')
                        pynn.Projection(self.layers[idxLayer][idxInput], layerDense, pynn.FromListConnector(synapseInhib), receptor_type='inhibitory')
                        self.synapsesNum += len(synapseExcit)+len(synapseInhib)
                else:
                    neuronNumInp = inputRow
                    weights = layer.weights
                    weightsMask = self._kernelMask(weights, layer.distribution, filterCode)
                    weights *= weightsMask

                    synapseExcit, synapseInhib = [], []
                    for io in range(neuronNumOut):
                        for ii in range(neuronNumInp):
                            weight = weights[io, ii]
                            if weight > 0:
                                synapseExcit.append((ii, io, weight, 1.0))
                            elif weight < 0:
                                synapseInhib.append((ii, io, weight, 1.0))

                    pynn.Projection(self.layers[idxLayer][0], layerDense, pynn.FromListConnector(synapseExcit), receptor_type='excitatory')
                    pynn.Projection(self.layers[idxLayer][0], layerDense, pynn.FromListConnector(synapseInhib), receptor_type='inhibitory')
                    self.synapsesNum += len(synapseExcit)+len(synapseInhib)

                self.layers.append([layerDense])
                layersShape.append({'shape': (neuronNumOut, 0), 'neuronNum': neuronNumOut})
            idxLayer += 1

    def plot_spike_train(self):
        # plot spike train in input of SNN
        import matplotlib.pyplot as plt
        plt.eventplot(self.spike)
        plt.show()

    def start_simulation(self, numberSamples, timeStimulus):
        layers = [layer[0] for layer in self.layers]
        [layer.record('spikes') for layer in layers]

        pynn.run((timeStimulus['duration']+timeStimulus['silence'])*numberSamples)

        spikeTrain = [layer.get_data().segments[0].spiketrains for layer in layers]

        pynn.end()
        return spikeTrain
