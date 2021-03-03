from operator import itemgetter
from random import randint

import mlrose_hiive as mlrose

from utils import get_six_peaks_problem, plot_maxima

import itertools

import numpy as np


def six_peaks_maxima(bitstring_length=15):
    fitness = mlrose.SixPeaks(t_pct=0.1)
    enumerations = [
        [int(x) for x in seq]
        for seq in itertools.product("01", repeat=bitstring_length)
    ]
    results = []
    for element in enumerations:
        results.append((element, fitness.evaluate(element)))
    print(sorted(results, reverse=True, key=itemgetter(1))[:100])
    plot_maxima(
        f"SixPeaks-Distribution-{bitstring_length}", [result[1] for result in results]
    )


def one_max_maxima(bitstring_length=15):
    fitness = mlrose.OneMax()
    enumerations = [
        [int(x) for x in seq]
        for seq in itertools.product("01", repeat=bitstring_length)
    ]
    results = []
    for element in enumerations:
        results.append((element, fitness.evaluate(element)))
    print(sorted(results, reverse=True, key=itemgetter(1))[:100])
    plot_maxima(
        f"OneMax-Distribution-{bitstring_length}", [result[1] for result in results]
    )


def flip_flop_maxima(bitstring_length=15):
    fitness = mlrose.FlipFlop()
    enumerations = [
        [int(x) for x in seq]
        for seq in itertools.product("01", repeat=bitstring_length)
    ]
    results = []
    for element in enumerations:
        results.append((element, fitness.evaluate(element)))
    print(sorted(results, reverse=True, key=itemgetter(1))[:100])
    plot_maxima(
        f"FlipFlop-Distribution-{bitstring_length}", [result[1] for result in results]
    )


def knapsack_maxima(bitstring_length=15):
    """
    Fitness function for Knapsack optimization problem.
    Given a set of n items, where item i has known weight w_{i} and known value v_{i};
    and maximum knapsack capacity, W,
    the Knapsack fitness function evaluates the fitness of a state vector x = [x_{0}, x_{1}, \ldots, x_{n-1}] as:

    Fitness(x) = \sum_{i = 0}^{n-1}v_{i}x_{i}, \text{ if}
    \sum_{i = 0}^{n-1}w_{i}x_{i} \leq W, \text{ and 0, otherwise,}

    where x_{i} denotes the number of copies of item i included in the knapsack.
    Fitness function for Knapsack optimization problem.

    """
    weights = [randint(1, bitstring_length // 2) for x in list(range(bitstring_length))]
    values = [randint(1, bitstring_length) for x in list(range(bitstring_length))]
    fitness = mlrose.Knapsack(weights=weights, values=values)
    enumerations = [
        np.array([int(x) for x in seq])
        for seq in itertools.product("01", repeat=bitstring_length)
    ]
    results = []
    for element in enumerations:
        results.append((element, fitness.evaluate(element)))
    print(sorted(results, reverse=True, key=itemgetter(1))[:100])
    plot_maxima(
        f"Knapsack-Maxima-{bitstring_length}", [result[1] for result in results]
    )


if __name__ == "__main__":
    flip_flop_maxima()
    six_peaks_maxima()
    knapsack_maxima()
