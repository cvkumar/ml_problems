import mlrose
import numpy as np

if __name__ == "__main__":
    bitstring_length = 16
    initial_state = np.array([0] * bitstring_length)

    fitness = mlrose.OneMax()
    problem = mlrose.DiscreteOpt(
        length=bitstring_length, fitness_fn=fitness, maximize=True, max_val=2
    )

    # Population size is the number of elements to keep alive during the algorithm's move forward
    # mutation probability is the probability of changing the child's individual state elements by one during
    # reproduction with the parents
    # TODO: Justify pop size and mutation prob parameter choices
    best_state, best_fitness, curve = mlrose.genetic_alg(
        problem=problem, pop_size=10, mutation_prob=0.1, max_iters=np.inf, curve=True
    )

    print(best_state)
    print(best_fitness)
    print(curve)
