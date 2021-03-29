"""
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

"""
import random

import pandas as pd

import math

RANDOM_STATE = 12
random.seed(RANDOM_STATE)

df = pd.read_csv("data/seeds/wheat-seeds.csv")

# print(df.shape)


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = []
    # TODO: Think about these layers more
    hidden_layer = [
        {"weights": [random.random() for i in range(n_inputs + 1)]}
        for i in range(n_hidden)
    ]
    network.append(hidden_layer)
    output_layer = [
        {"weights": [random.random() for i in range(n_hidden + 1)]}
        for i in range(n_outputs)
    ]
    network.append(output_layer)
    return network


def activate(weights, inputs):
    activation = weights[-1] # This is the bias value
    for i in range(len(inputs) - 1):
        activation += weights[i] * inputs[i]
    return activation


def transfer_neuron_activation(activation):
    # SIGMOID
    return 1.0 / (1.0 + math.exp(activation))


def forward_propagate(network, inputs):
    """
    given inputs

    """
    # TODO: There's a bug here to fix
    layer_values = inputs.copy()
    for layer in network:
        temp = []
        for neuron in layer:
            activation_val = transfer_neuron_activation(activate(weights=neuron['weights'], inputs=layer_values))
            temp.append(activation_val)

        layer_values = temp

    return layer_values


network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]

row = [1, 0, None]  # last one is bias
output = forward_propagate(network, row)
print(output)



# activation = sum(weight_i * input_i) + bias

# for layer in network:
#     print(layer)
