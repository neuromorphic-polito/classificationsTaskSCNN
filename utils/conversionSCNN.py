import os
from pygenn import GeNNModel, init_postsynaptic, init_weight_update
import numpy as np


class SNN:
    def __init__(self, device, lif, typology, dataset, modelCNN, reduction):
        #################################
        # ##### Backend selection ##### #
        #################################
        backend = None
        if 'cpu' in device.type:
            backend = 'single_threaded_cpu'
        elif 'cuda' in device.type:
            os.environ['CUDA_PATH'] = '/usr/local/cuda'
            backend = 'cuda'

        ################################
        # ##### Model definition ##### #
        ################################
        # ##### Model setting ##### #
        model = GeNNModel(precision='float', model_name='.model', backend=backend)
        model.dt = 1.0  # ms

        # ##### Neuron parameters ##### #
        lifParam, lifVar = lif

        # ##### Network definition ##### #
        layerPop, layerSyn = [], []

        timeStim = None
        if 'trainSetSpike' in typology:
            timeStim = dataset.trainSetSpike
        elif 'testSetSpike' in typology:
            timeStim = dataset.testSetSpike
        timeEnd = np.cumsum([len(spike) for spike in timeStim])
        timeStart = np.concatenate(([0], timeEnd[:-1]))
        timeSpike = np.concatenate(timeStim)

        # ##### Neuron populations ##### #
        sizeInRow, sizeInCol = dataset.shape
        popStim = model.add_neuron_population(
            pop_name=f'stim',
            num_neurons=sizeInRow*sizeInCol,
            neuron='SpikeSourceArray',
            params={}, vars={"startSpike": timeStart, "endSpike": timeEnd}
        )
        popStim.extra_global_params["spikeTimes"].set_init_values(timeSpike)
        popStim.spike_recording_enabled = True
        layerPop.append([popStim])

        flatFlag = False
        self.synapsesNum = 0
        for name, param in modelCNN.named_modules():
            if 'conv' in name:
                layerPop.append([])

                # ##### Neuron populations ##### #
                chOut, chIn, kernelRow, kernelCol = param.weight.shape
                sizeOutRow, sizeOutCol = sizeInRow-kernelRow+1, sizeInCol-kernelCol+1

                for o in range(chOut):
                    layerPop[-1].append(
                        model.add_neuron_population(
                            pop_name=f'pop{name}{o}',
                            num_neurons=sizeOutRow*sizeOutCol,
                            neuron='LIF',
                            params=lifParam, vars=lifVar,
                        )
                    )

                # ##### Synaptic connections ##### #
                connectionBase = []
                for iRow in range(sizeOutRow):
                    for iCol in range(sizeOutCol):
                        idxTarget = iCol+iRow*sizeOutCol
                        for kRow in range(kernelRow):
                            for kCol in range(kernelCol):
                                idxSource = (iRow+kRow)*sizeInCol+iCol+kCol
                                connectionBase.append([idxSource, idxTarget])
                connectionBase = np.array(connectionBase)

                weights = param.weight.detach().cpu().numpy()
                if reduction > 0:
                    threshold = np.quantile(np.abs(weights), reduction/100)
                    weights = np.where(np.abs(weights)>threshold, weights, 0.0)

                for i in range(chIn):
                    for o in range(chOut):
                        popSource = layerPop[-2][i]
                        popTarget = layerPop[-1][o]

                        weight = np.tile(weights[o, i].flatten(), sizeOutRow*sizeOutCol).reshape((-1, 1))
                        connection = np.hstack([connectionBase, weight])
                        connection = connection[connection[:, -1] != 0]

                        synPre = connection[:, 0].astype(int)
                        synPos = connection[:, 1].astype(int)
                        weight = connection[:, 2]
                        self.synapsesNum += weight.size
                        if weight.size > 0:
                            layerSyn.append(
                                model.add_synapse_population(
                                    pop_name=f'syn{name}_{i}_{o}', matrix_type='SPARSE',
                                    source=popSource, target=popTarget,
                                    postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                                    weight_update_init=init_weight_update('StaticPulse', {}, {'g': weight}),
                                )
                            )
                            layerSyn[-1].set_sparse_connections(synPre, synPos)
                sizeInRow, sizeInCol = sizeOutRow, sizeOutCol
            elif 'pool' in name:
                layerPop.append([])

                # ##### Neuron populations ##### #
                ch, kernel = len(layerPop[-2]), param.kernel_size
                sizeOutRow, sizeOutCol = sizeInRow//kernel, sizeInCol//kernel

                for o in range(ch):
                    layerPop[-1].append(
                        model.add_neuron_population(
                            pop_name=f'pop{name}{o}',
                            num_neurons=sizeOutRow*sizeOutCol,
                            neuron='LIF',
                            params=lifParam, vars=lifVar,
                        )
                    )

                # ##### Synaptic connections ##### #
                connection = []
                for iRow in range(sizeOutRow):
                    for iCol in range(sizeOutCol):
                        idxTarget = iCol+iRow*sizeOutCol
                        for kRow in range(kernel):
                            for kCol in range(kernel):
                                idxSource = iCol*kernel+kCol+(iRow*kernel+kRow)*sizeInCol
                                connection.append([idxSource, idxTarget])
                connection = np.array(connection)

                for i in range(ch):
                    popSource = layerPop[-2][i]
                    popTarget = layerPop[-1][i]

                    synPre = connection[:, 0]
                    synPos = connection[:, 1]
                    weight = np.tile(np.ones(kernel**2)/(kernel**2), sizeOutRow*sizeOutCol)
                    self.synapsesNum += weight.size
                    layerSyn.append(
                        model.add_synapse_population(
                            pop_name=f'syn{name}{i}', matrix_type='SPARSE',
                            source=popSource, target=popTarget,
                            postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                            weight_update_init=init_weight_update('StaticPulse', {}, {'g': weight}),
                        )
                    )
                    layerSyn[-1].set_sparse_connections(synPre, synPos)
                sizeInRow, sizeInCol = sizeOutRow, sizeOutCol
            elif 'flat' in name:
                flatFlag = True
            elif 'linear' in name:
                layerPop.append([])

                # ##### Neuron populations ##### #
                sizeIn, sizeOut = param.in_features, param.out_features

                layerPop[-1].append(
                    model.add_neuron_population(
                        pop_name=f'pop{name}',
                        num_neurons=sizeOut,
                        neuron='LIF',
                        params=lifParam, vars=lifVar,
                    )
                )

                weights = param.weight.detach().cpu().numpy()
                if reduction > 0:
                    threshold = np.quantile(np.abs(weights), reduction/100)
                    weights = np.where(np.abs(weights) > threshold, weights, 0.0)

                if flatFlag is True:
                    # ##### Synaptic connections ##### #
                    for i in range(len(layerPop[-2])):
                        popSource = layerPop[-2][i]
                        popTarget = layerPop[-1][0]

                        weight = weights[:, i*(sizeInRow*sizeInCol):(i+1)*(sizeInRow*sizeInCol)].T.flatten()
                        self.synapsesNum += weight.size
                        layerSyn.append(
                            model.add_synapse_population(
                                pop_name=f'syn{name}{i}', matrix_type='DENSE',
                                source=popSource, target=popTarget,
                                postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                                weight_update_init=init_weight_update('StaticPulse', {}, {'g': weight}),
                            )
                        )
                    flatFlag = False
                else:
                    # ##### Synaptic connections ##### #
                    popSource = layerPop[-2][0]
                    popTarget = layerPop[-1][0]

                    weight = weights.T.flatten()
                    self.synapsesNum += weight.size
                    layerSyn.append(
                        model.add_synapse_population(
                            pop_name=f'syn{name}', matrix_type='DENSE',
                            source=popSource, target=popTarget,
                            postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                            weight_update_init=init_weight_update('StaticPulse', {}, {'g': weight}),
                        )
                    )

        layerPop[-1][-1].spike_recording_enabled = True
        self.model = model
        self.layerOutput = layerPop[-1][-1]
