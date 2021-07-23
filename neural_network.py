from neural_network_functions import Function
from typing import Sequence
import random


class Layer:

    def __init__(self, neurons_n: int, connexions: int, act_f: Function, bias=None, weights=None):

        self.bias = []
        self.neurons_n = neurons_n
        self.connexions = connexions
        self.weights = []
        self.act_f = act_f

        self.raw_values = [None] * neurons_n
        self.computed_values = [None] * neurons_n
        self.deltas = [None] * neurons_n

        if bias is not None:
            if len(bias) != neurons_n:
                raise ValueError("len(bias) != neurons_n")
            self.bias = bias

        # Check if weights has correct format
        if weights is not None:
            if len(weights) == neurons_n:
                for i in range(neurons_n):
                    if len(weights[i]) != connexions:
                        raise ValueError("len(weights[{}]) != connexions".format(i))
            else:
                raise ValueError("len(weights) != neurons_n")

            self.weights = weights

        if bias is None or weights is None:

            for i in range(neurons_n):
                if bias is None:
                    self.bias.append(random.uniform(-1, 1))

                if weights is None:
                    self.weights.append([random.uniform(-1, 1) for i in range(connexions)])

    def compute_values(self, _input: Sequence):

        # _input parameter is the output of previous layer

        if len(_input) != self.connexions:
            raise ValueError("len(input) != connexions")

        for neuron_n in range(self.neurons_n):
            self.raw_values[neuron_n] = 0
            for connexion_n in range(self.connexions):
                self.raw_values[neuron_n] += self.weights[neuron_n][connexion_n] * _input[connexion_n]
            self.raw_values[neuron_n] += self.bias[neuron_n]  # TODO -> ASEGURAR SI ES + O - O SI DA LO MISMO

            self.computed_values[neuron_n] = self.act_f.normal(self.raw_values[neuron_n])

    def __str__(self):
        return '-----------\nbias: {}\nneurons_n: {}\nconnexions: {}\nweights: {}'.format(
            self.bias, self.neurons_n, self.connexions, self.weights)


class InitialLayer(Layer):

    def __init__(self, neurons_n):
        connexions = 1
        weights = [[1] for i in range(neurons_n)]
        bias = [0 for i in range(neurons_n)]

        act_f = Function(lambda x: x, lambda x: 1)

        super().__init__(neurons_n, connexions, act_f, bias=bias, weights=weights)

    def compute_values(self, _input: Sequence):

        # The initial layer has different compute_values func because values passed as
        # parameter are the input of Neural Network

        if len(_input) != self.neurons_n:
            raise ValueError("On initial layer, len(input) must be equals to neurons_n")

        self.computed_values = _input


class NeuralNetwork:

    def __init__(self, topology: Sequence, act_f: Function, cost_f: Function):

        # topology parameter is a list that contains neurons number of each layer
        # act_f parameter must be a function or a list of functions

        self.input_layer = InitialLayer(topology[0])

        self.layers = [self.input_layer]

        custom_act_f_for_each_layer = hasattr(act_f, '__iter__')
        if custom_act_f_for_each_layer and len(act_f) != len(topology) - 1:
            raise ValueError('len(act_f) != n_layers - 1')

        for i in range(1, len(topology)):
            if custom_act_f_for_each_layer:
                self.layers.append(Layer(topology[i], self.layers[i - 1].neurons_n, act_f[i - 1]))
            else:
                self.layers.append(Layer(topology[i], self.layers[i - 1].neurons_n, act_f))

        self.output_layer = self.layers[-1]

        self.cost_f = cost_f

        self.historic = None

    def get_output(self, _input: Sequence):

        for i in range(len(self.layers)):
            if i == 0:
                self.input_layer.compute_values(_input)
            else:
                self.layers[i].compute_values(self.layers[i-1].computed_values)

        return self.output_layer.computed_values

    def train(self, dataset: Sequence, expected_output: Sequence, times: int, learning_rate: float):

        if len(dataset) != len(expected_output):
            raise ValueError('len(dataset) != len(expected_output)')

        dataset_len = len(dataset)

        self.historic = []

        for rep in range(times):

            hist = []

            for data_index in range(dataset_len):

                output = self.get_output(dataset[data_index])
                hist.append(abs(self.cost_f.normal(expected_output[data_index][0], output[0])))

                deltas = [None for i in range(len(self.layers))]

                saved_weights = [layer.weights for layer in self.layers]

                # BACKPROPAGATION
                for layer_index in range(len(self.layers) - 1, -1, -1):

                    layer = self.layers[layer_index]

                    deltas[layer_index] = [None for layer_index in range(layer.neurons_n)]

                    # COMPUTE GRADIENT IN OUTPUT LAYER
                    for neuron_index in range(layer.neurons_n):
                        if layer is self.output_layer:

                            neuron_error = self.cost_f.normal(expected_output[data_index][neuron_index],
                                                              output[neuron_index])
                        else:

                            neuron_error = [saved_weights[layer_index + 1][n][neuron_index] * deltas[layer_index+1][n]
                                            for n in range(self.layers[layer_index+1].neurons_n)]
                            neuron_error = sum(neuron_error) / len(neuron_error)

                        derived_act = layer.act_f.derived(layer.computed_values[neuron_index])

                        deltas[layer_index][neuron_index] = neuron_error * derived_act

                        # GRADIENT DESCENT
                        if layer is not self.input_layer:

                            delta_mod = deltas[layer_index][neuron_index]*learning_rate

                            layer.bias[neuron_index] = layer.bias[neuron_index] + delta_mod

                            prev_layer_output = self.layers[layer_index-1].computed_values
                            layer.weights[neuron_index] = [layer.weights[neuron_index][n] + prev_layer_output[n] *
                                                           delta_mod for n in range(layer.connexions)]

            self.historic.append(sum(hist) / len(hist))

    def __str__(self):
        s = ''
        for layer in self.layers:
            s += str(layer) + '\n'
        return s
