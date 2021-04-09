"""
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

"""
import random

import pandas as pd

import math

RANDOM_STATE = 13
random.seed(RANDOM_STATE)

df = pd.read_csv("data/seeds/wheat-seeds.csv")


# print(df.shape)
# Number of weights, number of neurons
# [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]


class Neuron:
    def __init__(self, weights, bias=None):
        self.bias = bias or random.random()
        self.weights = weights

    def __str__(self):
        return f"weights: {self.weights}, bias: {self.bias}"


class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        # Add a neuron for n_hidden with random weights that can handle the number of inputs
        self.hidden_layer = [
            Neuron(bias=random.random(), weights=self._generate_weights(n_inputs))
            for i in range(n_hidden)
        ]

        # Add a neuron for each output with random weights that can handle the number of hidden layer neurons
        self.output_layer = [
            Neuron(bias=random.random(), weights=self._generate_weights(n_hidden))
            for i in range(n_outputs)
        ]

    def _generate_weights(self, n_weights: int) -> list:
        return [random.random() for i in range(n_weights)]

    def activate(self, weights, inputs, bias):
        """
        Basically given a neuron and inputs, compute the result of activating the neuron
        """
        # This is literally regression x)
        result = bias
        for i in range(len(weights)):
            result += inputs[i] * weights[i]
        return result

    def transfer_activation_value(self, activation: float):
        # SIGMOID
        return 1 / (1.0 + math.exp(-activation))

    def forward_propogate(self, inputs: list):
        hidden_layer_outputs = []
        for neuron in self.hidden_layer:
            activation_value = self.activate(weights=neuron.weights, inputs=inputs, bias=neuron.bias)
            neuron_output = self.transfer_activation_value(activation_value)
            hidden_layer_outputs.append(neuron_output)

        final_outputs = []
        for neuron in self.output_layer:
            activation_value = self.activate(weights=neuron.weights, inputs=hidden_layer_outputs, bias=neuron.bias)
            neuron_output = self.transfer_activation_value(activation_value)
            final_outputs.append(neuron_output)

        return final_outputs

    def __str__(self):
        hidden_layer = f"hidden_layer: {[str(neuron) for neuron in self.hidden_layer]}"
        output_layer = f"output_layer: {[str(neuron) for neuron in self.output_layer]}"
        return f"{hidden_layer}\n{output_layer}"


network = NeuralNetwork(n_inputs=2, n_hidden=2, n_outputs=2)
print(network)
print("\n")

sample_input = [1, 0]
result = network.forward_propogate(sample_input)
print(result)



