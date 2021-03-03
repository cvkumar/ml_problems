import numpy as np

from frozen_lake import FrozenLakeEnv
from policy_iteration import PolicyIteration
from q_learning import QLearning
from util import plot_policies
from value_iteration import ValueIteration


def run_value_iteration(
    env,
    max_iterations=1000,
    gamma=0.9,
    theta=1e-20,
    plot_interval=10,
    max_steps=200,
    n_episodes=10000,
    time_it=False,
):
    if time_it:
        plot_interval = 0
    vi = ValueIteration(
        env,
        max_iterations=max_iterations,
        gamma=gamma,
        theta=theta,
        hanoi=False,
        max_steps=max_steps,
        plot_interval=plot_interval,
    )
    vi.run()
    wins, total_reward, average_reward = play_episodes(
        env, n_episodes, vi.policy, max_steps
    )
    print(f"Value Iteration :: number of wins over {n_episodes} episodes = {wins}")
    print(
        f"Value Iteration :: average reward over {n_episodes} episodes = {average_reward} \n\n"
    )
    print(
        f"Value Iteration :: average reward over {n_episodes} episodes = {average_reward} \n\n"
    )
    print(f"Value Iteration policy grid")
    vi.print_game_grid(vi.policy)
    return vi


def run_policy_iteration(
    env,
    gamma=0.8,
    theta=1e-20,
    max_iterations=1000,
    debug=False,
    plot_interval=10,
    max_steps=200,
    n_episodes=10000,
    time_it=False,
):
    if time_it:
        plot_interval = 0
    pi = PolicyIteration(
        env,
        gamma=gamma,
        theta=theta,
        max_iterations=max_iterations,
        debug=debug,
        hanoi=False,
        plot_interval=plot_interval,
        max_steps=max_steps,
    )
    pi.policy_iteration()
    policy = [i[0] for i in pi.policy]
    wins, total_reward, average_reward = play_episodes(
        env, n_episodes, policy, max_steps
    )
    print(f"Policy Iteration :: number of wins over {n_episodes} episodes = {wins}")
    print(
        f"Policy Iteration :: average reward over {n_episodes} episodes = {average_reward} \n\n"
    )
    print(
        f"Policy Iteration :: average reward over {n_episodes} episodes = {average_reward} \n\n"
    )
    pi.print_game_grid(policy)
    return pi


def print_game_grid(flat_arr, length_of_board=8):
    for i in range(len(flat_arr)):
        print(flat_arr[i], end=" ")
        if (i + 1) % length_of_board == 0:
            print("\n")


def play_episodes(environment, n_episodes, policy, max_steps):
    wins = 0
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
            if terminated and reward > 0:
                wins += 1
            step += 1
    average_reward = total_reward / n_episodes
    return wins, total_reward, average_reward


def play_episodes_q_learning(environment, n_episodes, q_table, max_steps, debug=False):
    wins = 0
    losses = 0
    total_reward = 0
    for episode in range(n_episodes):
        terminated = False
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
            if terminated and reward == 1:
                if debug:
                    print(f"episode: {episode} - won at step: {step} in state {state}")
                wins += 1
            elif terminated:
                if debug:
                    print(
                        f"episode: {episode} - terminated early at step: {step} in state {state}"
                    )
                    environment.render()
                losses += 1
            step += 1
        if step >= max_steps:
            if debug:
                print(
                    f"episode: {episode} - took too long to find the end after {step} steps in state {state}"
                )
                environment.render()
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
    run_vi = True
    run_pi = True
    run_ql = True
    WINDY = False
    env = FrozenLakeEnv()
    vi = None
    pi = None
    q_learner = None
    time_it = False
    plot_interval = 0
    n_episodes = 10000

    # VALUE ITERATION
    if run_vi:
        max_iterations = 1000
        gamma = 0.99
        theta = 1e-20
        if not time_it:
            plot_interval = 10
        vi_env = env
        vi_env.reset()

        print(
            f"value iteration: gamma: {gamma}, theta: {theta}, max_iterations: {max_iterations}, plot_interval: {plot_interval}"
        )

        vi = run_value_iteration(
            vi_env,
            max_iterations,
            gamma,
            theta,
            plot_interval,
            n_episodes=n_episodes,
            time_it=time_it,
        )
        vi_env.render()

    # POLICY ITERATION
    if run_pi:
        gamma = 0.85
        theta = 1e-20
        max_iterations = 10000
        debug = False
        plot_interval = 1
        pi_env = env
        pi_env.reset()

        print(
            f"policy iteration: gamma: {gamma}, theta: {theta}, max_iterations: {max_iterations}, plot_interval: {plot_interval}"
        )
        pi = run_policy_iteration(
            pi_env,
            gamma,
            theta,
            max_iterations,
            debug,
            plot_interval,
            n_episodes=n_episodes,
            time_it=time_it,
        )
        pi_env.render()

    # Q-LEARNING
    if run_ql:
        ql_env = env
        ql_env.reset()

        plot_interval = 1000
        plot_threshold = 30000
        time_it = False
        if time_it:
            plot_interval = 0

        max_steps = 200
        n_episodes = 10000
        q_learner = QLearning(
            ql_env,
            max_episodes=500000,
            learning_rate=0.8,
            max_steps=max_steps,
            gamma=0.9,
            epsilon=1.0,
            min_epsilon=0.001,
            max_epsilon=1,
            decay_rate=0.0001,
            seed=1,
            threshold=0.8,
            rolling_window_size=300,
            plot_interval=plot_interval,
            plot_threshold=plot_threshold,
            debug=True,
        )
        q_learner.run()
        wins, losses, total_reward, average_reward = play_episodes_q_learning(
            ql_env, n_episodes, q_learner.q_table, max_steps
        )
        print(
            f"q_learner: episodes: {q_learner.max_episodes}, learning_rate: {q_learner.learning_rate}, "
            f"max_steps: {q_learner.max_steps}, gamma: {q_learner.gamma}, epsilon: {q_learner.epsilon},"
            f"min_epsilon: {q_learner.min_epsilon}, max_epsilon: {q_learner.max_epsilon}, "
            f"decay_rate: {q_learner.decay_rate}"
        )
        print(f"Q Learning :: number of wins over {n_episodes} episodes = {wins}")
        print(f"Q Learning :: number of losses over {n_episodes} episodes = {losses}")
        print(
            f"Q Learning :: average reward over {n_episodes} episodes = {average_reward} \n\n"
        )

    plot_policies(env, vi, pi, q_learner, hanoi=False)
