"""
sources

https://deepai.org/machine-learning-glossary-and-terms/softmax-layer
- Gives prob distribution given vector of real numbers.
- Sigmoid is kinda the same except with just one
- It is common to train a machine learning model using the softmax but switch out the softmax layer for an argmax layer when the model is used for inference.
- We must use softmax in training because the softmax is differentiable and it allows us to optimize a cost function.
- Because the softmax is a continuously differentiable function, it is possible to calculate the derivative of the loss function with respect to every weight in the network
"""

from typing import List
import math


def compute_softmax(vector: List[float]) -> List[float]:
    """

    """
    list_e_x = [math.exp(e_x) for e_x in vector]
    total = sum(list_e_x)
    return [e_x / total for e_x in list_e_x]


if __name__ == "__main__":
    print(compute_softmax([8, 5, 0]))
