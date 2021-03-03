import time

import matplotlib.pyplot as plt
import mlrose_hiive as mlrose


def get_one_max_problem(bitstring_length=20):
    fitness = mlrose.OneMax()
    return mlrose.DiscreteOpt(
        length=bitstring_length, fitness_fn=fitness, maximize=True, max_val=2
    )


def get_flip_flop_problem(bitstring_length=20):
    fitness = mlrose.FlipFlop()
    return mlrose.DiscreteOpt(
        length=bitstring_length, fitness_fn=fitness, maximize=True, max_val=2
    )


def get_six_peaks_problem(bitstring_length=20):
    fitness = mlrose.SixPeaks(t_pct=0.1)
    return mlrose.DiscreteOpt(
        length=bitstring_length, fitness_fn=fitness, maximize=True, max_val=2
    )


def get_knapsack_problem(
    weights=[10, 5, 2, 8, 15], values=[1, 2, 3, 4, 5], max_weight_pct=0.6
):
    fitness = mlrose.Knapsack(weights, values, max_weight_pct)
    return mlrose.DiscreteOpt(
        length=len(weights), fitness_fn=fitness, maximize=True, max_val=2
    )


def plot_fitness_iterations(
    title,
    iterations,
    sa_results,
    rhc_results,
    ga_results,
    mimic_results,
    seed="",
    file_name=None,
):
    if sa_results and len(sa_results) > 0:
        sa_line = plt.plot(iterations, sa_results, color="r", label="sa_fitness")
    if rhc_results and len(rhc_results) > 0:
        rhc_line = plt.plot(iterations, rhc_results, color="g", label="rhc_fitness")
    if ga_results and len(ga_results) > 0:
        ga_line = plt.plot(iterations, ga_results, color="b", label="ga_fitness")
    if mimic_results and len(mimic_results) > 0:
        mimic_line = plt.plot(
            iterations, mimic_results, color="y", label="mimic_fitness"
        )

    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Score")
    plt.legend()
    if not file_name:
        file_name = f"SixPeaks-{time.time()}-{seed}.png"
    plt.savefig(file_name)
    plt.close()


def plot_runtime_iterations(
    title,
    iterations,
    mimic_iterations,
    sa_time_results,
    rhc_time_results,
    ga_time_results,
    mimic_time_results,
    file_name,
):
    if sa_time_results and len(sa_time_results) == len(iterations):
        sa_time_line = plt.plot(
            iterations, sa_time_results, color="r", label="sa_runtime"
        )
    if rhc_time_results and len(rhc_time_results) == len(iterations):
        rhc_time_line = plt.plot(
            iterations, rhc_time_results, color="g", label="rhc_runtime"
        )
    if ga_time_results and len(ga_time_results) == len(iterations):
        ga_time_line = plt.plot(
            iterations, ga_time_results, color="b", label="ga_runtime"
        )
    if mimic_time_results and len(mimic_time_results) == len(iterations):
        mimic_time_line = plt.plot(
            mimic_iterations, mimic_time_results, color="y", label="mimic_runtime"
        )

    plt.title(title)
    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Log Runtime (seconds)")
    plt.legend()
    plt.savefig(file_name)
    plt.close()


def plot_maxima(title, results):
    plt.plot(results)
    plt.ylabel("fitness score")
    plt.xlabel("Bitstring Change")
    plt.savefig(title)
    plt.close()
