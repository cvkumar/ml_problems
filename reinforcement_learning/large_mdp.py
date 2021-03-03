import numpy as np

from policy_iteration import PolicyIteration
from q_learning import QLearning
from toh_env import TohEnv
from util import plot_policies
from value_iteration import ValueIteration


def run_value_iteration(
    env,
    n_episodes,
    max_steps,
    max_iterations=10000,
    gamma=0.9,
    theta=1e-20,
    plot_interval=10,
    time_it=False,
):
    if time_it:
        plot_interval = 0
    vi = ValueIteration(
        env,
        max_iterations=max_iterations,
        gamma=gamma,
        theta=theta,
        hanoi=True,
        plot_interval=plot_interval,
        max_steps=max_steps,
    )
    vi.run()
    wins, losses, total_reward, average_reward = play_episodes(
        env, n_episodes, vi.policy, max_steps
    )
    print(f"Value Iteration :: number of wins over {n_episodes} episodes = {wins}")
    print(f"Value Iteration :: number of losses over {n_episodes} episodes = {losses}")
    print(
        f"Value Iteration :: average reward over {n_episodes} episodes = {average_reward} \n\n"
    )
    return vi


def run_policy_iteration(
    env,
    n_episodes,
    max_steps,
    gamma=0.9,
    theta=1e-20,
    max_iterations=10000,
    debug=True,
    plot_interval=1,
    time_it=False,
):
    if time_it:
        plot_interval = 0
    pi = PolicyIteration(
        env,
        gamma=gamma,
        theta=theta,
        debug=debug,
        max_iterations=max_iterations,
        hanoi=True,
        plot_interval=plot_interval,
    )
    pi.policy_iteration()
    policy = [i[0] for i in pi.policy]
    wins, losses, total_reward, average_reward = play_episodes(
        env, n_episodes, policy, max_steps
    )
    print(f"Policy Iteration :: number of wins over {n_episodes} episodes = {wins}")
    print(f"Policy Iteration :: number of losses over {n_episodes} episodes = {losses}")
    print(
        f"Policy Iteration :: average reward over {n_episodes} episodes = {average_reward} \n\n"
    )
    return pi


def play_episodes(environment, n_episodes, policy, max_steps):
    wins = 0
    losses = 0
    total_reward = 0
    for episode in range(n_episodes):
        terminated = False
        state = environment.reset()
        step = 0
        while not terminated and step < max_steps:
            # Select best action to perform in a current state
            action = policy[state]
            # Perform an action an observe how environment acted in response
            next_state, reward, terminated, info = environment.step(action)
            # Summarize total reward
            total_reward += reward
            # Update current state
            state = next_state
            # Calculate number of wins over episodes
            if terminated and reward > 1.0:
                print(f"episode: {episode} - won at step: {step} in state {state}")
                wins += 1
            elif terminated:
                print(
                    f"episode: {episode} - terminated early at step: {step} in state {state}"
                )
                losses += 1
            step += 1
        if step >= max_steps:
            print(
                f"episode: {episode} - took too long to find the end after {step} steps in state {state}"
            )
            losses += 1
    average_reward = total_reward / n_episodes
    return (
        wins,
        losses,
        total_reward,
        average_reward,
    )


def play_episodes_q_learning(environment, n_episodes, q_table, max_steps, debug=False):
    wins = 0
    losses = 0
    total_reward = 0
    for episode in range(n_episodes):
        terminated = False
        # environment.seed(seed)
        state = environment.reset()
        step = 0
        while not terminated and step < max_steps:
            # Select best action to perform in a current state
            action = np.argmax(q_table[state, :])
            # Perform an action an observe how environment acted in response
            next_state, reward, terminated, info = environment.step(action)
            # Summarize total reward
            total_reward += reward
            # Update current state
            state = next_state
            # Calculate number of wins over episodes
            if terminated and reward > 1.0:
                if debug:
                    print(f"episode: {episode} - won at step: {step} in state {state}")
                wins += 1
            elif terminated:
                if debug:
                    print(
                        f"episode: {episode} - terminated early at step: {step} in state {state}"
                    )
                    print(environment.all_states[state])
                losses += 1
            step += 1
        if step >= max_steps:
            if debug:
                print(
                    f"episode: {episode} - took too long to find the end after {step} steps in state {state}"
                )
                print(environment.all_states[state])
                print("action was")
                print(action)
            losses += 1
    average_reward = total_reward / n_episodes
    return (
        wins,
        losses,
        total_reward,
        average_reward,
    )


if __name__ == "__main__":
    # initial_state = ((6, 5, 4, 3, 2, 1, 0), (), ())
    # goal_state = ((), (), (6, 5, 4, 3, 2, 1, 0))  # env.all_states[2144]
    initial_state = ((5, 4, 3, 2, 1, 0), (), ())
    goal_state = ((), (), (5, 4, 3, 2, 1, 0))  # env.all_states[686]
    # initial_state = ((3, 2, 1, 0), (), ())
    # goal_state = ((), (), (3, 2, 1, 0))
    """
    Optimal params for 4 things
    q_learner = QLearning(
    ql_env, episodes=10000, learning_rate=.8, max_steps=10000, gamma=.9, epsilon=1.0,
    min_epsilon=0.001, max_epsilon=1, decay_rate=0.0004
    )
    """
    # initial_state = ((2, 1, 0), (), ())
    # goal_state = ((), (), (2, 1, 0))
    noise = 0.01
    n_episodes = 10000
    max_steps = 5000
    vi = None
    pi = None
    q_learner = None
    run_vi = False
    run_pi = False
    run_ql = True
    debug = False
    time_it = False

    # VALUE ITERATION
    if run_vi:
        max_iterations = 10000
        gamma = 0.99
        theta = 1e-20
        plot_interval = 1
        vi_env = TohEnv(initial_state=initial_state, goal_state=goal_state, noise=noise)
        print(vi_env.all_states)
        print(
            f"value iteration: gamma: {gamma}, theta: {theta}, max_iterations: {max_iterations}"
        )
        vi = run_value_iteration(
            vi_env,
            n_episodes,
            max_steps,
            max_iterations,
            gamma,
            theta,
            plot_interval,
            time_it=time_it,
        )

    # POLICY ITERATION
    if run_pi:
        seed = 12
        gamma = 0.99
        theta = 1e-10
        max_iterations = 100000
        plot_interval = 1
        pi_env = TohEnv(initial_state=initial_state, goal_state=goal_state, noise=noise)
        pi_env.seed(seed)
        # gammas = [i for i in np.arange(0.99, 1, .0001)]
        # for gamma in gammas:
        print(
            f"policy iteration: gamma: {gamma}, theta: {theta}, max_iterations: {max_iterations}, plot_interval: {plot_interval}"
        )
        pi = run_policy_iteration(
            pi_env,
            n_episodes,
            max_steps,
            gamma,
            theta,
            max_iterations,
            True,
            0,
            time_it=time_it,
        )

    # Q-LEARNING
    if run_ql:
        seed = 7
        ql_env = TohEnv(initial_state=initial_state, goal_state=goal_state, noise=noise)
        ql_env.seed(seed)
        max_steps = 200
        time_it = False
        plot_interval = 1000
        plot_threshold = 10000  # 30000
        if time_it:
            plot_interval = 0

        q_learner = QLearning(
            ql_env,
            max_episodes=100000,
            min_episodes=30000,
            learning_rate=0.8,
            max_steps=max_steps,
            gamma=0.9,
            epsilon=1.0,
            min_epsilon=0.001,
            max_epsilon=1,
            decay_rate=0.00004,
            seed=seed,
            debug=True,
            rolling_window_size=500,
            threshold=80,
            plot_interval=plot_interval,
            plot_threshold=plot_threshold,
            hanoi=True,
        )
        q_learner.run()
        wins, losses, total_reward, average_reward = play_episodes_q_learning(
            ql_env, n_episodes, q_learner.q_table, max_steps
        )
        print(
            f"noise: {noise}, q_learner: episodes: {q_learner.max_episodes}, learning_rate: {q_learner.learning_rate}, "
            f"max_steps: {q_learner.max_steps}, gamma: {q_learner.gamma}, epsilon: {q_learner.epsilon},"
            f"min_epsilon: {q_learner.min_epsilon}, max_epsilon: {q_learner.max_epsilon}, "
            f"decay_rate: {q_learner.decay_rate}"
        )
        print(f"Q Learning :: number of wins over {n_episodes} episodes = {wins}")
        print(f"Q Learning :: number of losses over {n_episodes} episodes = {losses}")
        print(
            f"Q Learning :: average reward over {n_episodes} episodes = {average_reward} \n\n"
        )

    plot_policies(vi_env, vi, pi, q_learner, hanoi=True)
