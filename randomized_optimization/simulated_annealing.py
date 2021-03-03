import mlrose
import numpy as np

if __name__ == "__main__":
    bitstring_length = 16
    initial_state = np.array([0] * bitstring_length)

    fitness = mlrose.OneMax()
    problem = mlrose.DiscreteOpt(
        length=bitstring_length, fitness_fn=fitness, maximize=True, max_val=2
    )

    # As time goes on (over iterations), the number produced by the schedule decreases.
    # This indicates there is less likelihood of it getting selected.
    # TODO: How do I choose these parameters?
    schedule = mlrose.GeomDecay(init_temp=1, decay=0.9, min_temp=0.001)

    # TODO: Justify max_attempts equal to two
    best_state, best_fitness, curve = mlrose.simulated_annealing(
        problem=problem,
        max_attempts=2,
        max_iters=np.inf,
        init_state=initial_state,
        curve=True,
        random_state=42,
    )

    print(best_state)
    print(best_fitness)
    print(curve)
