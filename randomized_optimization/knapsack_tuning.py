from mlrose_hiive import MIMICRunner, GARunner, SARunner, RHCRunner

from asgn2_randomized_optimization.knapsack import get_knapsack_problem

"""
best fitness parameters:
      Iteration  Fitness      Time  ... Population Size  Keep Percent  max_iters
1021         10     32.0  0.043126  ...              30           0.6       4910
1022        110     32.0  0.142627  ...              30           0.6       4910
1023        210     32.0  0.142627  ...              30           0.6       4910
1024        310     32.0  0.142627  ...              30           0.6       4910
1025        410     32.0  0.142627  ...              30           0.6       4910
...         ...      ...       ...  ...             ...           ...        ...
7696       4510     32.0  0.318204  ...             190           0.8       4910
7697       4610     32.0  0.318204  ...             190           0.8       4910
7698       4710     32.0  0.318204  ...             190           0.8       4910
7699       4810     32.0  0.318204  ...             190           0.8       4910
7700       4910     32.0  0.318204  ...             190           0.8       4910
"""


def tune_mimic(problem, experiment_name, print_output=True, return_tuned_params=False):
    mmc = MIMICRunner(
        problem=problem,
        experiment_name=experiment_name,
        output_directory="mimic",
        seed=42,
        iteration_list=list(range(1, 100)),
        max_attempts=5,
        keep_percent_list=[0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9],
        population_sizes=list(range(1, 100, 10)),
        use_fast_mimic=True,
    )
    # the two data frames will contain the results
    df_run_stats, df_run_curves = mmc.run()
    best_results = df_run_stats[df_run_stats.Fitness == max(df_run_stats.Fitness)]
    if True:
        return best_results
    if print_output:
        print(
            f"best fitness parameters:\n{df_run_stats[df_run_stats.Fitness == max(df_run_stats.Fitness)]}"
        )
    if not return_tuned_params:
        return df_run_stats, df_run_curves
    else:
        best_params = {}
        return df_run_stats, df_run_curves, best_params


"""
best fitness parameters:
      Iteration  Fitness      Time  ... Population Size  Mutation Rate  max_iters
1            10     32.0  0.013006  ...              10            0.1         95
2            15     32.0  0.023343  ...              10            0.1         95
3            20     32.0  0.033555  ...              10            0.1         95
4            25     32.0  0.043016  ...              10            0.1         95
5            30     32.0  0.052860  ...              10            0.1         95
...         ...      ...       ...  ...             ...            ...        ...
2047         75     32.0  0.417110  ...              95            0.6         95
2048         80     32.0  0.417110  ...              95            0.6         95
2049         85     32.0  0.417110  ...              95            0.6         95
2050         90     32.0  0.417110  ...              95            0.6         95
2051         95     32.0  0.417110  ...              95            0.6         95
"""


def tune_ga(problem, experiment_name, print_output=True, return_tuned_params=False):
    ga = GARunner(
        problem=problem,
        experiment_name=experiment_name,
        output_directory="ga",
        seed=42,
        iteration_list=range(10, 100, 5),
        max_attempts=25,
        population_sizes=list(range(10, 100, 5)),
        mutation_rates=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )

    # the two data frames will contain the results
    df_run_stats, df_run_curves = ga.run()
    best_results = df_run_stats[df_run_stats.Fitness == max(df_run_stats.Fitness)]
    if True:
        return best_results
    if print_output:
        print(f"best fitness parameters:\n{best_results}")
    if not return_tuned_params:
        return df_run_stats, df_run_curves
    else:
        #  TODO:
        best_params = {}
        return df_run_stats, df_run_curves, best_params


def tune_rhc(problem, experiment_name, print_output=True):
    rhc = RHCRunner(
        problem=problem,
        experiment_name=experiment_name,
        output_directory="rhc",
        seed=42,
        iteration_list=range(10, 100, 10),
        max_attempts=100,
        restart_list=list(range(10, 100, 10)),
    )

    # the two data frames will contain the results
    df_run_stats, df_run_curves = rhc.run()
    best_results = df_run_stats[df_run_stats.Fitness == max(df_run_stats.Fitness)]
    if True:
        return best_results
    if print_output:
        print(f"best fitness parameters:\n{best_results}")
    return df_run_stats, df_run_curves


def tune_sa(problem, experiment_name, print_output=True, return_tuned_params=False):
    sa = SARunner(
        problem=problem,
        experiment_name=experiment_name,
        output_directory="sa",
        seed=42,
        iteration_list=range(10, 100, 20),
        max_attempts=100,
        temperature_list=[0.1, 0.2, 0.3, 0.4, 0.5, 1, 10, 50, 100],
    )

    # the two data frames will contain the results
    df_run_stats, df_run_curves = sa.run()
    best_results = df_run_stats[df_run_stats.Fitness == max(df_run_stats.Fitness)]
    if True:
        return best_results
    if print_output:
        print(f"best fitness parameters:\n{best_results}")
    if not return_tuned_params:
        return df_run_stats, df_run_curves
    else:
        best_params = {"init_temp": best_results["schedule_init_temp"].iloc[0]}
        print(f"best sa kwargs:\n{best_params}")
        return df_run_stats, df_run_curves, best_params


if __name__ == "__main__":
    BITSTRING_LENGTH = 20
    # weights = [12, 3, 4, 7, 25, 20, 40, 32, 17, 21]
    # KNAPSACK_VALUES =  [1, 2, 3, 4, 5, 6, 7, 4, 2, 5]
    KNAPSACK_WEIGHTS = [2, 12, 3, 4, 7, 25, 20, 40, 32, 17]
    KNAPSACK_VALUES = [1, 2, 3, 4, 5, 6, 7, 4, 2, 5]
    # weights = [2, 12, 3, 4, 7, 25, 20, 40, 32, 17, 2, 52, 3, 4, 13, 7, 29, 40, 57, 2]
    # values = [1, 2, 3, 4, 5, 6, 7, 4, 2, 5, 2, 12, 23, 4, 54, 1, 3, 9, 24, 12]

    # sample weights and values
    # weights = [23, 26, 20, 18, 32, 27, 29, 26, 30, 27]
    # values = [505, 352, 458, 220, 354, 414, 498, 545, 473, 543]
    MAX_WEIGHT_PCT = 0.6
    problem = get_knapsack_problem(KNAPSACK_WEIGHTS, KNAPSACK_VALUES, MAX_WEIGHT_PCT)

    experiment_name = "knapsack_tuning"

    """
    Tune parameters for knapsack problem
    """
    # rhc = tune_rhc(problem, experiment_name)
    mimic = tune_mimic(problem, experiment_name)
    # ga = tune_ga(problem, experiment_name)
    # sa = tune_sa(problem, experiment_name)
    # print(f"rhc:\n{rhc}\n")
    print(f"mimic:\n{mimic}\n")
    # print(f"ga:\n{ga}\n")
    # print(f"sa:\n{sa}\n")
