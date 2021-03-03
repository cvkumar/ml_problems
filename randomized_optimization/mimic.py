import mlrose
import numpy as np

if __name__ == "__main__":
    bitstring_length = 16
    initial_state = np.array([0] * bitstring_length)

    fitness = mlrose.OneMax()
    problem = mlrose.DiscreteOpt(
        length=bitstring_length, fitness_fn=fitness, maximize=True, max_val=2
    )

    # TODO: Justify chosen pop size and keep_pct
    best_state, best_fitness, curve = mlrose.mimic(
        problem=problem,
        pop_size=10,
        keep_pct=0.2,
        max_iters=np.inf,
        max_attempts=5,
        curve=True,
    )

    print(best_state)
    print(best_fitness)
    print(curve)
