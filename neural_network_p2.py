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
    def __init__(self, weights, bias=None, output=None):
        self.bias = bias or random.random()
        self.weights = weights
        self.delta = 0
        self.output = output or 0

    def __str__(self):
        return f"weights: {self.weights}, bias: {self.bias}, delta: {self.delta}, output: {self.output}"


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
            neuron.output = self.transfer_activation_value(activation_value)
            hidden_layer_outputs.append(neuron.output)

        final_outputs = []
        for neuron in self.output_layer:
            activation_value = self.activate(weights=neuron.weights, inputs=hidden_layer_outputs, bias=neuron.bias)
            neuron.output = self.transfer_activation_value(activation_value)
            final_outputs.append(neuron.output)

        return final_outputs

    def transfer_derivative(self, output):
        """
        output ~ Output of neuron
        Calculates derivative with respect to sigmoid transfer function

        # TODO: add comment for what delta
        """
        return output * (1.0 - output)

    def backward_propogate_error(self, expected):
        # for i in reversed(range(len(self.output_layer))):
        layer = self.output_layer
        errors = []

        for j in range(len(self.output_layer)):
            neuron = layer[j]
            errors.append(expected[j] - neuron.output)

        for j in range(len(self.output_layer)):
            neuron = layer[j]
            neuron.delta = errors[j] * self.transfer_derivative(neuron.output)

        for j in range(len(self.hidden_layer)):
            error = 0.0
            for neuron in self.output_layer:
                error += (neuron.weights[j] * neuron.delta)
            errors.append(error)

        for j in range(len(self.hidden_layer)):
            neuron = self.hidden_layer[j]
            neuron.delta = errors[j] * self.transfer_derivative(neuron.output)

        # print(errors)
        # for j in range(len(self.hidden_layer)):




    def __str__(self):
        hidden_layer = f"hidden_layer: {[str(neuron) for neuron in self.hidden_layer]}"
        output_layer = f"output_layer: {[str(neuron) for neuron in self.output_layer]}"
        return f"{hidden_layer}\n{output_layer}"

# from sample problem: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
# [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
# [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
network = NeuralNetwork(n_inputs=2, n_hidden=1, n_outputs=2)
network.hidden_layer = [Neuron(weights=[0.13436424411240122, 0.8474337369372327], bias=0.763774618976614)]
network.output_layer = [Neuron(weights=[0.2550690257394217], bias=0.49543508709194095), Neuron(weights=[0.4494910647887381], bias=0.651592972722763)]
print(network)
print("")

sample_input = [1, 0]
result = network.forward_propogate(sample_input)
print("Forward propogate result")
print(result)
# Should be [0.6629970129852887, 0.7253160725279748]
print("")

sample_expected_output = [0, 1]
"""
network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
"""
network.hidden_layer = [Neuron(weights=[0.13436424411240122, 0.8474337369372327], bias=0.763774618976614, output=0.7105668883115941)]
network.output_layer = [Neuron(weights=[0.2550690257394217], bias=0.49543508709194095, output=0.6213859615555266), Neuron(weights=[0.4494910647887381], bias=0.651592972722763, output=0.6573693455986976)]
network.backward_propogate_error(sample_expected_output)
print("network after back propogating")
print(network)
"""
[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614], 'delta': -0.0005348048046610517}]
[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095], 'delta': -0.14619064683582808}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763], 'delta': 0.0771723774346327}]
"""
print("")

