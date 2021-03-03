"""

Fitness function for Six Peaks optimization problem. Evaluates the fitness of an n-dimensional state vector x, given parameter T, as:

Fitness(x, T) = max(tail(0, x), head(1, x)) + R(x, T)

where:
tail(0, x) is the number of trailing 0’s in x;
head(1, x) is the number of leading 1’s in x;
R(x, T) = n, if (tail(0, x) > T and head(1, x) > T) or (tail(1, x) > T and head(0, x) > T); and
R(x, T) = 0, otherwise.

"""

import time
from random import randint

import mlrose_hiive as mlrose

from utils import plot_runtime_iterations, plot_fitness_iterations, get_knapsack_problem

import numpy as np

BITSTRING_LENGTH = 40
KNAPSACK_WEIGHTS = [
    randint(1, BITSTRING_LENGTH // 2) for x in list(range(BITSTRING_LENGTH))
]
KNAPSACK_VALUES = [randint(1, BITSTRING_LENGTH) for x in list(range(BITSTRING_LENGTH))]
MAX_WEIGHT_PCT = 0.6
PROBLEM_NAME = "KNAPSACK"
PROBLEM = get_knapsack_problem(KNAPSACK_WEIGHTS, KNAPSACK_VALUES, MAX_WEIGHT_PCT)
SA_RHC_MAX_ATTEMPTS = BITSTRING_LENGTH // 2
MIMIC_MAX_ATTEMPTS = BITSTRING_LENGTH // 1
MIMIC_POPULATION_SIZE = 75
MIMIC_KEEP_PERCENT = 0.6
GA_POPULATION_SIZE = 20
GA_MUTATION_RATE = 0.1
GA_MAX_ATTEMPTS = BITSTRING_LENGTH // 5
SA_TEMP = 0.1
RHC_RESTARTS = BITSTRING_LENGTH // 2
RANDOM_STATE = list(range(30, 61, 2))
ALGORITHMS_TO_RUN = ["rhc", "sa", "ga", "mimic"]

ITERATIONS = 200

"""
    # sample weights and values
    weights = [23, 26, 20, 18, 32, 27, 29, 26, 30, 27]
    values = [505, 352, 458, 220, 354, 414, 498, 545, 473, 543]
    MAX_WEIGHT_PCT = 0.6
    
rhc:
      Iteration  Fitness        Time  ... Restarts  max_iters  current_restart
3631         10   2901.0  132.077298  ...       80         90               76
3632         20   2901.0  132.297127  ...       80         90               76
3633         30   2901.0  132.518315  ...       80         90               76
3634         40   2901.0  132.745679  ...       80         90               76
3635         50   2901.0  132.964726  ...       80         90               76
3636         60   2901.0  133.184624  ...       80         90               76
3637         70   2901.0  133.403303  ...       80         90               76
3638         80   2901.0  133.623566  ...       80         90               76
3639         90   2901.0  133.845368  ...       80         90               76
4441         10   2901.0  170.469469  ...       90         90               76
4442         20   2901.0  170.754848  ...       90         90               76
4443         30   2901.0  171.022198  ...       90         90               76
4444         40   2901.0  171.286306  ...       90         90               76
4445         50   2901.0  171.551673  ...       90         90               76
4446         60   2901.0  171.816087  ...       90         90               76
4447         70   2901.0  172.082666  ...       90         90               76
4448         80   2901.0  172.347989  ...       90         90               76
4449         90   2901.0  172.605621  ...       90         90               76

[18 rows x 7 columns]

mimic:
      Iteration  Fitness      Time  ... Population Size  Keep Percent  max_iters
1786         10   3022.0  0.044445  ...              50           0.5       4910
1787        110   3022.0  0.138587  ...              50           0.5       4910
1788        210   3022.0  0.138587  ...              50           0.5       4910
1789        310   3022.0  0.138587  ...              50           0.5       4910
1790        410   3022.0  0.138587  ...              50           0.5       4910
...         ...      ...       ...  ...             ...           ...        ...
7747       4510   3022.0  0.396456  ...             190           0.9       4910
7748       4610   3022.0  0.396456  ...             190           0.9       4910
7749       4710   3022.0  0.396456  ...             190           0.9       4910
7750       4810   3022.0  0.396456  ...             190           0.9       4910
7751       4910   3022.0  0.396456  ...             190           0.9       4910

[2244 rows x 8 columns]

ga:
      Iteration  Fitness      Time  ... Population Size  Mutation Rate  max_iters
40           15   3022.0  0.018672  ...              10            0.3         95
41           20   3022.0  0.029333  ...              10            0.3         95
42           25   3022.0  0.039854  ...              10            0.3         95
43           30   3022.0  0.050289  ...              10            0.3         95
44           35   3022.0  0.059907  ...              10            0.3         95
...         ...      ...       ...  ...             ...            ...        ...
2047         75   3022.0  0.568023  ...              95            0.6         95
2048         80   3022.0  0.568023  ...              95            0.6         95
2049         85   3022.0  0.568023  ...              95            0.6         95
2050         90   3022.0  0.568023  ...              95            0.6         95
2051         95   3022.0  0.568023  ...              95            0.6         95

[1741 rows x 7 columns]

sa:
    Iteration  Fitness      Time  ... schedule_current_value Temperature  max_iters
1          10   2810.0  0.000989  ...               0.099999         0.1         90
2          30   2810.0  0.007615  ...               0.099992         0.1         90
3          50   2810.0  0.012978  ...               0.099987         0.1         90
4          70   2810.0  0.018504  ...               0.099981         0.1         90
5          90   2810.0  0.026158  ...               0.099974         0.1         90
7          10   2810.0  0.000954  ...               0.199998         0.2         90
8          30   2810.0  0.007706  ...               0.199985         0.2         90
9          50   2810.0  0.014375  ...               0.199971         0.2         90
10         70   2810.0  0.022746  ...               0.199954         0.2         90
11         90   2810.0  0.028925  ...               0.199942         0.2         90
13         10   2810.0  0.001145  ...               0.299997         0.3         90
14         30   2810.0  0.009208  ...               0.299972         0.3         90
15         50   2810.0  0.017346  ...               0.299948         0.3         90
16         70   2810.0  0.025602  ...               0.299923         0.3         90
17         90   2810.0  0.032504  ...               0.299902         0.3         90
19         10   2810.0  0.000954  ...               0.399996         0.4         90
20         30   2810.0  0.009536  ...               0.399962         0.4         90
21         50   2810.0  0.016847  ...               0.399932         0.4         90
22         70   2810.0  0.024796  ...               0.399900         0.4         90
23         90   2810.0  0.033091  ...               0.399867         0.4         90
25         10   2810.0  0.001290  ...               0.499994         0.5         90
26         30   2810.0  0.009984  ...               0.499950         0.5         90
27         50   2810.0  0.019147  ...               0.499904         0.5         90
28         70   2810.0  0.027601  ...               0.499861         0.5         90
29         90   2810.0  0.037241  ...               0.499813         0.5         90
31         10   2810.0  0.000902  ...               0.999991           1         90
32         30   2810.0  0.009252  ...               0.999907           1         90
33         50   2810.0  0.017919  ...               0.999820           1         90
34         70   2810.0  0.028182  ...               0.999717           1         90
35         90   2810.0  0.038541  ...               0.999613           1         90
37         10   2810.0  0.000873  ...               9.999912          10         90
38         30   2810.0  0.010583  ...               9.998936          10         90
39         50   2810.0  0.020748  ...               9.997915          10         90
40         70   2810.0  0.030862  ...               9.996899          10         90
41         90   2810.0  0.041925  ...               9.995787          10         90
43         10   2810.0  0.000881  ...              49.999557          50         90
44         30   2810.0  0.011332  ...              49.994306          50         90
45         50   2810.0  0.020667  ...              49.989616          50         90
46         70   2810.0  0.032048  ...              49.983898          50         90
47         90   2810.0  0.041509  ...              49.979145          50         90
"""


def _print_algo_results(name, fitness, state, result_curve=None):
    print(
        f"{name} results:\n best fitness score - {fitness}\n best state - {state}\n curve - {result_curve} \n"
    )


def find_average(number_list):
    return sum(number_list) / len(number_list)


def _average_result_per_iter(iter_to_results):
    avg_result = []
    for key, val in iter_to_results.items():
        avg_result.append(find_average(val))
    return avg_result


def _extend_result_curve(curve):
    for i in range(len(curve), ITERATIONS):
        curve = np.append(curve, curve[len(curve) - 1])
    return curve


if __name__ == "__main__":

    sa_iteration_to_results = [0] * ITERATIONS
    rhc_iteration_to_results = [0] * ITERATIONS
    ga_iteration_to_results = [0] * ITERATIONS
    mimic_iteration_to_results = [0] * ITERATIONS
    sa_curve, rhc_curve, mimic_curve, ga_curve = None, None, None, None

    print(f"Running algorithms for iteration: {ITERATIONS}")
    for seed in RANDOM_STATE:
        print(f"Running algorithms for seed: {seed}")
        sa_results, rhc_results, ga_results, mimic_results = [], [], [], []
        sa_time_results, rhc_time_results, ga_time_results, mimic_time_results = (
            [],
            [],
            [],
            [],
        )

        experiment_name = "example_experiment"

        """
        Simulated Annealing

                Iteration  Fitness      Time  ... schedule_current_value Temperature  max_iters
        15         80     62.0  0.086788  ...               0.099913         0.1         95
        16         85     62.0  0.092706  ...               0.099907         0.1         95
        17         90     62.0  0.097808  ...               0.099902         0.1         95
        18         95     62.0  0.102851  ...               0.099897         0.1         95
        34         80     62.0  0.091548  ...               0.199816         0.2         95
        35         85     62.0  0.097396  ...               0.199804         0.2         95
        36         90     62.0  0.104529  ...               0.199790         0.2         95
        37         95     62.0  0.113175  ...               0.199773         0.2         95
        53         80     62.0  0.135159  ...               0.299593         0.3         95
        54         85     62.0  0.142926  ...               0.299569         0.3         95
        55         90     62.0  0.150768  ...               0.299546         0.3         95
        56         95     62.0  0.160258  ...               0.299517         0.3         95
        72         80     62.0  0.157967  ...               0.399365         0.4         95
        73         85     62.0  0.169184  ...               0.399320         0.4         95
        74         90     62.0  0.180417  ...               0.399275         0.4         95
        75         95     62.0  0.188842  ...               0.399242         0.4         95
        """
        if "sa" in ALGORITHMS_TO_RUN:
            sa_name = "Simulated_Annealing"
            schedule = mlrose.GeomDecay(init_temp=SA_TEMP, decay=0.8, min_temp=0.001)
            sa_start_time = time.time()
            sa_state, sa_fitness, sa_curve = mlrose.simulated_annealing(
                problem=PROBLEM,
                max_attempts=SA_RHC_MAX_ATTEMPTS,
                max_iters=ITERATIONS,
                random_state=seed,
                curve=True,
            )
            # Extend curve all the way to iteration length
            sa_curve = _extend_result_curve(sa_curve)

            for i in range(len(sa_curve)):
                sa_iteration_to_results[i] += sa_curve[i]

            sa_end_time = time.time()
            sa_time_results.append(sa_end_time - sa_start_time)

        """
        Randomized Hill Climbing

        best fitness parameters:
                Iteration  Fitness        Time  ... Restarts  max_iters  current_restart
        2868         80     61.0  101.620677  ...       70         90               70
        2869         90     61.0  101.804123  ...       70         90               70
        3578         80     61.0  128.967706  ...       80         90               70
        3579         90     61.0  129.202814  ...       80         90               70
        4388         80     61.0  160.292064  ...       90         90               70
        4389         90     61.0  160.559458  ...       90         90               70
        """
        if "rhc" in ALGORITHMS_TO_RUN:
            rhc_name = "Random Hill Climbing"
            rhc_start_time = time.time()
            rhc_state, rhc_fitness, rhc_curve = mlrose.random_hill_climb(
                problem=PROBLEM,
                max_attempts=SA_RHC_MAX_ATTEMPTS,
                max_iters=ITERATIONS,
                restarts=RHC_RESTARTS,
                curve=True,
                random_state=seed,
            )
            rhc_curve = _extend_result_curve(rhc_curve)

            for i in range(len(rhc_curve)):
                rhc_iteration_to_results[i] += rhc_curve[i]

            rhc_end_time = time.time()
            rhc_time_results.append(rhc_end_time - rhc_start_time)

        """
        Genetic Algorithm

                Iteration  Fitness      Time  ... Population Size  Mutation Rate  max_iters
        889          80     94.0  0.964168  ...              45            0.5         95
        890          85     94.0  1.031187  ...              45            0.5         95
        891          90     94.0  1.095282  ...              45            0.5         95
        892          95     94.0  1.160694  ...              45            0.5         95
        985          85     94.0  1.116889  ...              50            0.4         95
        ...         ...      ...       ...  ...             ...            ...        ...
        2013         95     94.0  2.714904  ...              95            0.4         95
        2048         80     94.0  2.089107  ...              95            0.6         95
        2049         85     94.0  2.238257  ...              95            0.6         95
        2050         90     94.0  2.382583  ...              95            0.6         95
        2051         95     94.0  2.520674  ...              95            0.6         95
        """
        if "ga" in ALGORITHMS_TO_RUN:
            ga_name = "Genetic Algorithm"
            ga_start_time = time.time()
            ga_state, ga_fitness, ga_curve = mlrose.genetic_alg(
                problem=PROBLEM,
                pop_size=GA_POPULATION_SIZE,
                mutation_prob=GA_MUTATION_RATE,
                max_iters=ITERATIONS,
                curve=True,
                random_state=seed,
                max_attempts=GA_MAX_ATTEMPTS,
            )
            ga_curve = _extend_result_curve(ga_curve)

            for i in range(len(ga_curve)):
                ga_iteration_to_results[i] += ga_curve[i]

            ga_end_time = time.time()
            ga_time_results.append(ga_end_time - ga_start_time)

        """
        MIMIC

        mimic:
              Iteration  Fitness      Time  ... Population Size  Keep Percent  max_iters
        2903          3     33.0  0.070124  ...              31          0.75         99
        2904          4     33.0  0.103197  ...              31          0.75         99
        2905          5     33.0  0.136604  ...              31          0.75         99
        2906          6     33.0  0.169956  ...              31          0.75         99
        2907          7     33.0  0.203773  ...              31          0.75         99
        ...         ...      ...       ...  ...             ...           ...        ...
        7695         95     33.0  0.591141  ...              91          0.60         99
        7696         96     33.0  0.591141  ...              91          0.60         99
        7697         97     33.0  0.591141  ...              91          0.60         99
        7698         98     33.0  0.591141  ...              91          0.60         99
        7699         99     33.0  0.591141  ...              91          0.60         99
        """
        if "mimic" in ALGORITHMS_TO_RUN:
            mimic_name = "Mimic"
            mimic_start_time = time.time()
            PROBLEM.set_mimic_fast_mode(True)
            mimic_state, mimic_fitness, mimic_curve = mlrose.mimic(
                problem=PROBLEM,
                pop_size=MIMIC_POPULATION_SIZE,
                keep_pct=MIMIC_KEEP_PERCENT,
                max_iters=ITERATIONS,
                max_attempts=MIMIC_MAX_ATTEMPTS,
                curve=True,
                random_state=seed,
            )
            mimic_curve = _extend_result_curve(mimic_curve)

            for i in range(len(mimic_curve)):
                mimic_iteration_to_results[i] += mimic_curve[i]

            mimic_end_time = time.time()
            mimic_time_results.append(mimic_end_time - mimic_start_time)

    # print(f"seed: {seed} - sa best fitness: {max(sa_results)}")
    # print(f"seed: {seed} - sa max time taken: {max(sa_time_results)}")
    # print(f"seed: {seed} - ga max fitness: {max(ga_results)}")
    # print(f"seed: {seed} - ga max time taken: {max(ga_time_results)}")
    # print(f"seed: {seed} - rhc max fitness: {max(rhc_results)}")
    # print(f"seed: {seed} - rhc max time taken: {max(rhc_time_results)}")
    # print(f"seed: {seed} - mimic max fitness: {max(mimic_results)}")
    # print(f"seed: {seed} - mimic max time taken: {max(mimic_time_results)}")

    # PLOT RESULTS
    # plot_fitness_iterations("{} Fitness vs. Iterations".format(PROBLEM_NAME), ITERATIONS, MIMIC_ITERATIONS,
    #                         sa_results,
    #                         rhc_results, ga_results,
    #                         mimic_results, file_name="{}-individual-{}.png".format(PROBLEM_NAME, seed))

    # plot_runtime_iterations("{} Runtime vs. Iterations".format(PROBLEM_NAME), ITERATIONS, MIMIC_ITERATIONS,
    #                         sa_time_results,
    #                         rhc_time_results,
    #                         ga_time_results, mimic_time_results,
    #                         file_name="{}-runtime-{}.png".format(PROBLEM_NAME, time.time()))

    print(f"SA: Average time per seed:{find_average(sa_time_results)}")
    print(f"RHC: Average time per seed:{find_average(rhc_time_results)}")
    print(f"MIMIC: Average time per seed:{find_average(mimic_time_results)}")
    print(f"GA: Average time per seed:{find_average(ga_time_results)}")

    sa_avg_result = [i / len(RANDOM_STATE) for i in sa_iteration_to_results]
    rhc_avg_result = [i / len(RANDOM_STATE) for i in rhc_iteration_to_results]
    mimic_avg_result = [i / len(RANDOM_STATE) for i in mimic_iteration_to_results]
    ga_avg_result = [i / len(RANDOM_STATE) for i in ga_iteration_to_results]

    title = "Average {} Fitness vs. Iterations".format(PROBLEM_NAME)
    plot_fitness_iterations(
        title=title,
        iterations=list(range(ITERATIONS)),
        sa_results=sa_avg_result,
        rhc_results=rhc_avg_result,
        mimic_results=mimic_avg_result,
        ga_results=ga_avg_result,
        file_name="{}-average-{}.png".format(PROBLEM_NAME, time.time()),
    )
