import numpy as np


##########################
# CNN feature extraction #
##########################
class CNN:
    def __init__(self, model):
        self.layers, flag = [], False

        for layer in model.layers:
            if 'flatten' not in layer.name:
                self.layers.append(Layer(layer))
            elif 'flatten' in layer.name:
                flag = True
            if 'dense' in layer.name and flag:
                flag = False
                neuronInput = self.layers[-1].input
                neuronOutput = self.layers[-1].output
                featureOutput = self.layers[-3].featureOutput

                dimensionMap = int(neuronInput/featureOutput)  # dimension of single feature map

                weights = self.layers[-1].weights
                tmp = np.zeros((neuronOutput, neuronInput))

                for feature_map in range(featureOutput):
                    for cell in range(dimensionMap):
                        tmp[:, (cell+feature_map*dimensionMap)] = weights[:, feature_map+cell*featureOutput]
                self.layers[-1].weights = tmp


class Layer:
    def __init__(self, layer):
        if 'conv2d' in layer.name:
            shape = layer.get_weights()[0].shape

            self.type = 'conv2d'
            self.sizeKernel, self.featureInput, self.featureOutput = (shape[0], shape[1]), shape[2], shape[3]

            self.weights = []
            for idxOutput in range(self.featureOutput):
                tmp = []
                for idxInput in range(self.featureInput):
                    tmp.append(np.array(layer.get_weights()[0][:, :, idxInput, idxOutput]))
                self.weights.append(tmp)
            self.distribution = self._distribution(layer.get_weights()[0])

        elif 'max_pooling2d' in layer.name or 'average_pooling2d' in layer.name:
            self.type = 'pool2d'

            self.sizePool = layer.pool_size
            self.weights = np.ones(self.sizePool)*1/(self.sizePool[0]*self.sizePool[1])

        elif 'dense' in layer.name:
            shape = layer.get_weights()[0].shape

            self.type = 'dense'
            self.input, self.output = shape[0], shape[1]

            self.weights = np.transpose(layer.get_weights()[0])
            self.distribution = self._distribution(layer.get_weights()[0])


    def _distribution(self, weights):

        weightsAbs = np.abs(weights.flatten())

        q1 = np.quantile(weightsAbs, 0.25)
        q2 = np.quantile(weightsAbs, 0.50)
        q3 = np.quantile(weightsAbs, 0.75)
        M, m = q3+1.5*(q3-q1), q1-1.5*(q3-q1)
        W = np.max(np.delete(weightsAbs, np.argwhere(weightsAbs >= M)))
        w = np.min(np.delete(weightsAbs, np.argwhere(weightsAbs < m)))

        parameters = [0, w, q1, q2, q3, W]
        return parameters
