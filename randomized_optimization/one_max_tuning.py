import mlrose_hiive
from mlrose_hiive import MIMICRunner, GARunner, SARunner, RHCRunner

from asgn2_randomized_optimization.knapsack import get_knapsack_problem
from asgn2_randomized_optimization.six_peaks import get_six_peaks_problem
from utils import get_one_max_problem


def tune_mimic(problem, experiment_name, print_output=True, return_tuned_params=False):
    mmc = MIMICRunner(
        problem=problem,
        experiment_name=experiment_name,
        output_directory="mimic",
        seed=42,
        iteration_list=list(range(10, 5000, 100)),
        max_attempts=25,
        keep_percent_list=[0.1, 0.25, 0.4, 0.5, 0.6, 0.75],
        population_sizes=list(range(10, 100, 5)),
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
        iteration_list=range(10, 1000, 50),
        max_attempts=8,
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
        seed=3,
        iteration_list=range(10, 1000, 50),
        max_attempts=50,
        temperature_list=[0.1, 0.2, 0.3, 0.4, 0.5, 1, 5, 10, 50, 100, 1000],
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
    bitstring_length = 50
    problem = get_one_max_problem(bitstring_length)

    experiment_name = "example_experiment"

    # tune_rhc(problem, experiment_name)
    # tune_mimic(problem, experiment_name)
    # tune_ga(problem, experiment_name)
    tune_mimic(problem, experiment_name, return_tuned_params=True)
