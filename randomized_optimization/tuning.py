import mlrose_hiive
from mlrose_hiive import MIMICRunner, GARunner, SARunner, RHCRunner

from asgn2_randomized_optimization.knapsack import get_knapsack_problem
from asgn2_randomized_optimization.six_peaks import get_six_peaks_problem
from utils import get_flip_flop_problem


def tune_mimic(problem, experiment_name, print_output=True, return_tuned_params=False):
    mmc = MIMICRunner(
        problem=problem,
        experiment_name=experiment_name,
        output_directory="mimic",
        seed=42,
        iteration_list=list(range(10, 5000, 100)),
        max_attempts=25,
        keep_percent_list=[0.1, 0.25, 0.4, 0.5, 0.6, 0.75],
        population_sizes=list(range(10, 100, 10)),
        use_fast_mimic=True,
    )
    # the two data frames will contain the results
    df_run_stats, df_run_curves = mmc.run()
    if print_output:
        print(
            f"best fitness parameters:\n{df_run_stats[df_run_stats.Fitness == max(df_run_stats.Fitness)]}"
        )
    if not return_tuned_params:
        return df_run_stats, df_run_curves
    else:
        best_params = {}
        return df_run_stats, df_run_curves, best_params


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
        max_attempts=500,
        restart_list=list(range(10, 100, 10)),
    )

    # the two data frames will contain the results
    df_run_stats, df_run_curves = rhc.run()
    best_results = df_run_stats[df_run_stats.Fitness == max(df_run_stats.Fitness)]
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
        max_attempts=500,
        temperature_list=[0.1, 0.2, 0.3, 0.4, 0.5, 1, 10, 50, 100],
    )

    # the two data frames will contain the results
    df_run_stats, df_run_curves = sa.run()
    best_results = df_run_stats[df_run_stats.Fitness == max(df_run_stats.Fitness)]
    if print_output:
        print(f"best fitness parameters:\n{best_results}")
    if not return_tuned_params:
        return df_run_stats, df_run_curves
    else:
        best_params = {"init_temp": best_results["schedule_init_temp"].iloc[0]}
        print(f"best sa kwargs:\n{best_params}")
        return df_run_stats, df_run_curves, best_params


if __name__ == "__main__":
    peaks_problem = get_six_peaks_problem()
    knapsack_problem = get_knapsack_problem()
    flip_flop_problem = get_flip_flop_problem(50)

    experiment_name = "example_experiment"

    """
    Tune parameters for 6-peaks problem
    """
    # tune_rhc(peaks_problem, experiment_name)
    # tune_mimic(peaks_problem, experiment_name)
    # tune_ga(peaks_problem, experiment_name)
    # tune_sa(peaks_problem, experiment_name, return_tuned_params=True)

    """
    Tune parameters for knapsack problem
    """
    # tune_rhc(knapsack_problem, experiment_name)
    # tune_mimic(knapsack_problem, experiment_name)
    # tune_ga(knapsack_problem, experiment_name)
    # tune_sa(knapsack_problem, experiment_name)

    tune_sa(flip_flop_problem, experiment_name)
