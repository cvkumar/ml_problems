import mlrose
import numpy as np

if __name__ == "__main__":
    bitstring_length = 16
    initial_state = np.array([0] * bitstring_length)

    fitness = mlrose.OneMax()
    problem = mlrose.DiscreteOpt(
        length=bitstring_length, fitness_fn=fitness, maximize=True, max_val=2
    )
    best_state, best_fitness, curve = mlrose.random_hill_climb(
        problem=problem,
        max_attempts=5,
        max_iters=np.inf,
        restarts=1,
        init_state=initial_state,
        curve=True,
        random_state=42,
    )

    print(best_state)
    print(best_fitness)
    print(curve)
