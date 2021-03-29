"""
It's basically linear. It's differentiable. It avoids problem of vanishing gradient.



Vanishing gradient: https://machinelearningmastery.com/how-to-fix-vanishing-gradients-using-the-rectified-linear-activation-function/
Unable to propagate useful gradient information from the output end of the model back to the layers near the input end of the model

A problem with training networks with many layers (e.g. deep neural networks) is that
the gradient diminishes dramatically as it is propagated backward through the network.
The error may be so small by the time it reaches layers close to the input of the model
that it may have very little effect.

"""


def compute_relu(num: float) -> float:
    """

    """
    return max(0.0, num)


if __name__ == "__main__":
    print(compute_relu(-2))
