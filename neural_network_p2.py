"""
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

"""
import random

import pandas as pd

import math

RANDOM_STATE = 1
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
            activation_value = self.activate(
                weights=neuron.weights, inputs=inputs, bias=neuron.bias
            )
            neuron.output = self.transfer_activation_value(activation_value)
            hidden_layer_outputs.append(neuron.output)

        final_outputs = []
        for neuron in self.output_layer:
            activation_value = self.activate(
                weights=neuron.weights, inputs=hidden_layer_outputs, bias=neuron.bias
            )
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
        layer = self.output_layer
        output_layer_errors = []

        for j in range(len(self.output_layer)):
            neuron = layer[j]
            output_layer_errors.append(expected[j] - neuron.output)

        for j in range(len(self.output_layer)):
            neuron = layer[j]
            neuron.delta = output_layer_errors[j] * self.transfer_derivative(
                neuron.output
            )

        hidden_layer_errors = []
        for j in range(len(self.hidden_layer)):
            error = 0.0
            for neuron in self.output_layer:
                error += neuron.weights[j] * neuron.delta
            hidden_layer_errors.append(error)

        for j in range(len(self.hidden_layer)):
            neuron = self.hidden_layer[j]
            neuron.delta = hidden_layer_errors[j] * self.transfer_derivative(
                neuron.output
            )

    # NOTE: Mostly tested
    def update_weights(self, inputs, learning_rate):
        for neuron in self.hidden_layer:
            for j in range(len(inputs) - 1):
                neuron.weights[j] += learning_rate * neuron.delta * inputs[j]
            neuron.bias += learning_rate * neuron.delta

        inputs = [
            neuron.output for neuron in self.hidden_layer
        ]  # because hidden layer are inputs for output layer
        for neuron in self.output_layer:
            for j in range(len(inputs) - 1):
                neuron.weights[j] += learning_rate * neuron.delta * inputs[j]
            neuron.bias += learning_rate * neuron.delta

    # NOTE: Mostly tested
    def train_network(self, dataset, learning_rate, n_epoch, n_outputs):
        for epoch in range(n_epoch):
            sum_error = 0
            for inputs in dataset:
                outputs = self.forward_propogate(inputs)

                # Basically one hot encoding of output class
                expected = [0 for i in range(n_outputs)]
                expected[
                    inputs[-1]
                ] = 1  # If the class is 0 then make [0, 1] expected, otherwise [1, 0]

                # sum of squared errors
                sum_error += sum(
                    [(expected[i] - outputs[i]) ** 2 for i in range(len(expected))]
                )
                self.backward_propogate_error(expected=expected)
                self.update_weights(inputs, learning_rate)

            print(
                ">epoch=%d, lrate=%.3f, error=%.3f" % (epoch, learning_rate, sum_error)
            )

    def predict(self, inputs):
        outputs = self.forward_propogate(inputs=inputs)
        return outputs.index(max(outputs))

    def __str__(self):
        hidden_layer = f"hidden_layer: {[str(neuron) for neuron in self.hidden_layer]}"
        output_layer = f"output_layer: {[str(neuron) for neuron in self.output_layer]}"
        return f"{hidden_layer}\n{output_layer}"


if __name__ == "__main__":
    # # from sample problem: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    # # [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
    # # [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
    # network = NeuralNetwork(n_inputs=2, n_hidden=1, n_outputs=2)
    # network.hidden_layer = [
    #     Neuron(weights=[0.13436424411240122, 0.8474337369372327], bias=0.763774618976614)
    # ]
    # network.output_layer = [
    #     Neuron(weights=[0.2550690257394217], bias=0.49543508709194095),
    #     Neuron(weights=[0.4494910647887381], bias=0.651592972722763),
    # ]
    # print(network)
    # print("")
    #
    # sample_input = [1, 0]
    # result = network.forward_propogate(sample_input)
    # print("Forward propogate result")
    # print(result)
    # # Should be [0.6629970129852887, 0.7253160725279748]
    # print("")
    #
    # sample_expected_output = [0, 1]
    # """
    # network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
    # [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
    # """
    #
    # network.hidden_layer = [
    #     Neuron(
    #         weights=[0.13436424411240122, 0.8474337369372327],
    #         bias=0.763774618976614,
    #         output=0.7105668883115941,
    #     )
    # ]
    # network.output_layer = [
    #     Neuron(
    #         weights=[0.2550690257394217],
    #         bias=0.49543508709194095,
    #         output=0.6213859615555266,
    #     ),
    #     Neuron(
    #         weights=[0.4494910647887381], bias=0.651592972722763, output=0.6573693455986976
    #     ),
    # ]
    # network.backward_propogate_error(sample_expected_output)
    # print("network after back propogating")
    # print(network)
    # """
    # [{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614], 'delta': -0.0005348048046610517}]
    # [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095], 'delta': -0.14619064683582808}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763], 'delta': 0.0771723774346327}]
    # """
    # print("")

    dataset = [
        [2.7810836, 2.550537003, 0],
        [1.465489372, 2.362125076, 0],
        [3.396561688, 4.400293529, 0],
        [1.38807019, 1.850220317, 0],
        [3.06407232, 3.005305973, 0],
        [7.627531214, 2.759262235, 1],
        [5.332441248, 2.088626775, 1],
        [6.922596716, 1.77106367, 1],
        [8.675418651, -0.242068655, 1],
        [7.673756466, 3.508563011, 1],
    ]

    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    network = NeuralNetwork(n_inputs=n_inputs, n_hidden=2, n_outputs=n_outputs)
    network.train_network(
        dataset=dataset, learning_rate=5, n_epoch=100, n_outputs=n_outputs
    )
    print(network)
    """
    [{'weights': [-1.4688375095432327, 1.850887325439514, 1.0858178629550297], 'output': 0.029980305604426185, 'delta': -0.0059546604162323625}, {'weights': [0.37711098142462157, -0.0625909894552989, 0.2765123702642716], 'output': 0.9456229000211323, 'delta': 0.0026279652850863837}]
[{'weights': [2.515394649397849, -0.3391927502445985, -0.9671565426390275], 'output': 0.23648794202357587, 'delta': -0.04270059278364587}, {'weights': [-2.5584149848484263, 1.0036422106209202, 0.42383086467582715], 'output': 0.7790535202438367, 'delta': 0.03803132596437354}]
    """
    for inputs in dataset:
        prediction = network.predict(inputs)
        print("Expected=%d, Got=%d" % (inputs[-1], prediction))
