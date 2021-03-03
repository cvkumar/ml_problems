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

import mlrose_hiive as mlrose

from utils import (
    get_six_peaks_problem,
    plot_runtime_iterations,
    plot_fitness_iterations,
)

import numpy as np

PROBLEM_NAME = "SIX_PEAKS"
BITSTRING_LENGTH = 50
PROBLEM = get_six_peaks_problem(BITSTRING_LENGTH)
PLOT_PER_SEED = False
ITERATIONS = 1000

SA_RHC_MAX_ATTEMPTS = BITSTRING_LENGTH // 2  # For RHC and SA only
MIMIC_MAX_ATTEMPTS = BITSTRING_LENGTH // 4

# SA RHC: Function evaluations = iterations * max attempts * random_restarts
# Mimic: Function evaluations = iterations * max attempts * population
# GA: Function evaluations = iterations * max attempts * population

RHC_RESTARTS = 70

GA_MAX_ATTEMPTS = BITSTRING_LENGTH // 2

MIMIC_POPULATION_SIZE = 70
GA_POPULATION_SIZE = 45

RANDOM_STATE = list(range(61, 80, 1))
ALGORITHMS_TO_RUN = ["sa", "rhc", "mimic", "ga"]


def _print_algo_results(name, fitness, state, result_curve=None):
    print(
        f"{name} results:\n best fitness score - {fitness}\n best state - {state}\n curve - {result_curve} \n"
    )


def find_average(number_list):
    if number_list:
        return sum(number_list) / len(number_list)


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
            schedule = mlrose.GeomDecay(init_temp=1, decay=0.8, min_temp=0.001)
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
            # Extend curve all the way to iteration length
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
                mutation_prob=0.5,
                max_iters=ITERATIONS,
                curve=True,
                random_state=seed,
                max_attempts=GA_MAX_ATTEMPTS,
            )
            # Extend curve all the way to iteration length
            ga_curve = _extend_result_curve(ga_curve)

            for i in range(len(ga_curve)):
                ga_iteration_to_results[i] += ga_curve[i]

            ga_end_time = time.time()
            ga_time_results.append(ga_end_time - ga_start_time)

        """
        MIMIC

        best fitness parameters:
            Iteration  Fitness      Time  ... Population Size  Keep Percent  max_iters
        400         10     67.0  0.155060  ...              80          0.25         95
        401         15     67.0  0.256587  ...              80          0.25         95
        402         20     67.0  0.351296  ...              80          0.25         95
        403         25     67.0  0.450894  ...              80          0.25         95
        404         30     67.0  0.543232  ...              80          0.25         95
        405         35     67.0  0.635126  ...              80          0.25         95
        406         40     67.0  0.737539  ...              80          0.25         95
        407         45     67.0  0.829894  ...              80          0.25         95
        408         50     67.0  0.920735  ...              80          0.25         95
        409         55     67.0  1.018379  ...              80          0.25         95
        410         60     67.0  1.110209  ...              80          0.25         95
        411         65     67.0  1.204443  ...              80          0.25         95
        412         70     67.0  1.304133  ...              80          0.25         95
        413         75     67.0  1.396062  ...              80          0.25         95
        414         80     67.0  1.486271  ...              80          0.25         95
        415         85     67.0  1.581182  ...              80          0.25         95
        416         90     67.0  1.676411  ...              80          0.25         95
        417         95     67.0  1.762827  ...              80          0.25         95
        
        second run with more iterations: 
        best fitness parameters:
             Iteration  Fitness      Time  ... Population Size  Keep Percent  max_iters
        199         10     35.0  0.138363  ...              70          0.25       4510
        200        510     35.0  0.401105  ...              70          0.25       4510
        201       1010     35.0  0.401105  ...              70          0.25       4510
        202       1510     35.0  0.401105  ...              70          0.25       4510
        203       2010     35.0  0.401105  ...              70          0.25       4510
        204       2510     35.0  0.401105  ...              70          0.25       4510
        205       3010     35.0  0.401105  ...              70          0.25       4510
        206       3510     35.0  0.401105  ...              70          0.25       4510
        207       4010     35.0  0.401105  ...              70          0.25       4510
        208       4510     35.0  0.401105  ...              70          0.25       4510

        best fitness parameters:
              Iteration  Fitness      Time  ... Population Size  Keep Percent  max_iters
        1939         10     36.0  0.071208  ...              70           0.4       4910
        1940        110     36.0  0.238436  ...              70           0.4       4910
        1941        210     36.0  0.238436  ...              70           0.4       4910
        1942        310     36.0  0.238436  ...              70           0.4       4910
        1943        410     36.0  0.238436  ...              70           0.4       4910
        1944        510     36.0  0.238436  ...              70           0.4       4910
        1945        610     36.0  0.238436  ...              70           0.4       4910
        1946        710     36.0  0.238436  ...              70           0.4       4910
        1947        810     36.0  0.238436  ...              70           0.4       4910
        1948        910     36.0  0.238436  ...              70           0.4       4910
        1949       1010     36.0  0.238436  ...              70           0.4       4910
        1950       1110     36.0  0.238436  ...              70           0.4       4910
        1951       1210     36.0  0.238436  ...              70           0.4       4910
        1952       1310     36.0  0.238436  ...              70           0.4       4910
        1953       1410     36.0  0.238436  ...              70           0.4       4910
        1954       1510     36.0  0.238436  ...              70           0.4       4910
        1955       1610     36.0  0.238436  ...              70           0.4       4910
        1956       1710     36.0  0.238436  ...              70           0.4       4910
        1957       1810     36.0  0.238436  ...              70           0.4       4910
        1958       1910     36.0  0.238436  ...              70           0.4       4910
        1959       2010     36.0  0.238436  ...              70           0.4       4910
        1960       2110     36.0  0.238436  ...              70           0.4       4910
        1961       2210     36.0  0.238436  ...              70           0.4       4910
        1962       2310     36.0  0.238436  ...              70           0.4       4910
        1963       2410     36.0  0.238436  ...              70           0.4       4910
        1964       2510     36.0  0.238436  ...              70           0.4       4910
        1965       2610     36.0  0.238436  ...              70           0.4       4910
        1966       2710     36.0  0.238436  ...              70           0.4       4910
        1967       2810     36.0  0.238436  ...              70           0.4       4910
        1968       2910     36.0  0.238436  ...              70           0.4       4910
        1969       3010     36.0  0.238436  ...              70           0.4       4910
        1970       3110     36.0  0.238436  ...              70           0.4       4910
        1971       3210     36.0  0.238436  ...              70           0.4       4910
        1972       3310     36.0  0.238436  ...              70           0.4       4910
        1973       3410     36.0  0.238436  ...              70           0.4       4910
        1974       3510     36.0  0.238436  ...              70           0.4       4910
        1975       3610     36.0  0.238436  ...              70           0.4       4910
        1976       3710     36.0  0.238436  ...              70           0.4       4910
        1977       3810     36.0  0.238436  ...              70           0.4       4910
        1978       3910     36.0  0.238436  ...              70           0.4       4910
        1979       4010     36.0  0.238436  ...              70           0.4       4910
        1980       4110     36.0  0.238436  ...              70           0.4       4910
        1981       4210     36.0  0.238436  ...              70           0.4       4910
        1982       4310     36.0  0.238436  ...              70           0.4       4910
        1983       4410     36.0  0.238436  ...              70           0.4       4910
        1984       4510     36.0  0.238436  ...              70           0.4       4910
        1985       4610     36.0  0.238436  ...              70           0.4       4910
        1986       4710     36.0  0.238436  ...              70           0.4       4910
        1987       4810     36.0  0.238436  ...              70           0.4       4910
        1988       4910     36.0  0.238436  ...              70           0.4       4910
        """
        if "mimic" in ALGORITHMS_TO_RUN:
            mimic_name = "Mimic"
            mimic_start_time = time.time()
            PROBLEM.set_mimic_fast_mode(True)
            mimic_state, mimic_fitness, mimic_curve = mlrose.mimic(
                problem=PROBLEM,
                pop_size=MIMIC_POPULATION_SIZE,
                keep_pct=0.4,
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
        # if PLOT_PER_SEED:
        #     plot_fitness_iterations("{} Fitness vs. Iterations".format(PROBLEM_NAME), ITERATIONS,
        #                             sa_curve,
        #                             rhc_curve, ga_curve,
        #                             mimic_curve, seed)

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
