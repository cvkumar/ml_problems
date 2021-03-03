import time

import matplotlib.pyplot as plt
import numpy as np

from toh_env import TohEnv
from util import calculate_bellman_equation

"""
    Based on pseudo code from http://incompleteideas.net/book/RLbook2018.pdf page 75
    https://github.com/llSourcell/navigating_a_virtual_world_with_dynamic_programming/blob/master/dp/dp.py
    and https://towardsdatascience.com/reinforcement-learning-demystified-solving-mdps-with-dynamic-programming-b52c8093c919
"""


class PolicyIteration:
    def __init__(
        self,
        env,
        gamma=0.9,
        theta=1e-20,
        max_iterations=1000,
        debug=False,
        hanoi=False,
        plot_interval=0,
        max_steps=200,
    ):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.debug = debug
        self.hanoi = hanoi
        self.policy = [[0] for i in range(self.env.nS)]
        self.state_utilities = [0 for i in range(self.env.nS)]
        self.plot_interval = plot_interval
        self.max_steps = max_steps
        self.convergence = 0

        self.time_to_convergence = None

    def print_game_grid(self, flat_arr, length_of_board=8):
        # 0: up, 1: right, 2: down, 3: left
        directions = ["↑", "→", "↓", "←"]

        for i in range(len(flat_arr)):
            print(directions[flat_arr[i]], end=" ")
            if (i + 1) % length_of_board == 0:
                print("\n")

    def one_step_lookahead(self, state_index, v):
        """
        Helper function to calculate a state value function

        :param state_index: index of the current state to consider
        :param v: value to use as an estimator (vector of length nS)
        :return: expected value of each action for the given state (vector of length nA)
        """
        action_values = np.zeros(self.env.nA)
        for action_index in range(self.env.nA):
            action_values[action_index] += calculate_bellman_equation(
                self.env, action_index, state_index, v, self.gamma
            )
        return action_values

    def policy_evaluation(self, policy, plot=False):
        V = [0 for i in range(self.env.nS)]
        deltas = []
        mean_Vs = []
        for i in range(int(self.max_iterations)):
            # if i % 10 == 0 and self.debug:
            #     print(f"evaluation: {i}")
            delta = 0
            for state_index in range(self.env.nS):
                v_old = V[state_index]
                V[state_index] = 0
                if type(policy[state_index]) == list:
                    for action_index in policy[state_index]:
                        V[state_index] += calculate_bellman_equation(
                            self.env, action_index, state_index, V, self.gamma
                        )
                # else:
                #     action_index = policy[state_index]
                #     V[state_index] += calculate_bellman_equation(self.env, action_index, state_index, V, self.gamma)
                delta = max(delta, abs(v_old - V[state_index]))
            if plot:
                mean_V = np.mean(V[state_index])
            if plot:
                deltas.append(delta)
                mean_Vs.append(mean_V)

            if delta < self.theta:
                print(f"Policy evaluated in {i} iterations.")
                break
        if self.debug:
            print(f"Policy evaluated in {i} iterations")
            print(f"V: {V}")
            print(f"delta: {delta}")
        self.state_utilities = V
        return V

    def policy_iteration(self):
        print("----------------------- Policy Iteration Begin -----------------------")
        start_time = time.time()

        # initialize arbitrary policy and value-function to start
        policy = [[0] for i in range(self.env.nS)]
        intervals = []
        avg_rewards = []
        for i in range(int(self.max_iterations)):
            if i % 100 == 0 and self.debug:
                print(f"iteration: {i}")
            V = self.policy_evaluation(policy)
            policy_stable = True
            for state_index in range(self.env.nS):
                old_action = policy[state_index]
                actions = self.one_step_lookahead(state_index, V)
                policy[state_index] = [np.argmax(actions)]
                if old_action != policy[state_index]:
                    policy_stable = False

            if policy_stable:
                self.time_to_convergence = time.time() - start_time
                print(f"Evaluated {i} policies.")
                print(f"policy stable")
                p = [i[0] for i in policy]
                wins, total_reward, average_reward = self.play_episodes(p)
                print(
                    f"i: {i}, average reward: {average_reward}, time to converge: {self.time_to_convergence}"
                )
                intervals.append(i)
                avg_rewards.append(average_reward)
                self.convergence = i
                break

            if self.plot_interval > 0 and i % self.plot_interval == 0:
                p = [i[0] for i in policy]
                wins, total_reward, average_reward = self.play_episodes(p)
                print(f"i: {i}, average reward: {average_reward}")
                intervals.append(i)
                avg_rewards.append(average_reward)

        print(f"Evaluated {i} policies")
        if self.debug:
            print(f"state utilities: {V}")
            print(f"Policy: {policy}")

        # PLOT PERFORMANCE VS. ITERATION
        if self.plot_interval > 0:
            print("plotting value iteration")
            self._plot(intervals, avg_rewards)

        self.policy = policy
        self.state_utilities = V

    def play_episodes(self, policy, n_episodes=100, max_steps=200):
        wins = 0
        total_reward = 0
        for episode in range(n_episodes):
            terminated = False
            state = self.env.reset()
            step = 0
            while not terminated and step < max_steps:
                # Select best action to perform in a current state
                action = policy[state]
                # Perform an action an observe how environment acted in response
                next_state, reward, terminated, info = self.env.step(action)
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

    def _plot(self, xi, y):
        """
        xi is the intervals that we played the game
        y is average reward for n_episodes of playing the game at that interval
        """
        plt.rcParams["figure.figsize"] = (12, 6)
        fig, ax = plt.subplots()
        plt.plot(xi, y, marker="o", linestyle="--", color="b")

        plt.xlabel("Number of Iterations")
        plt.ylabel("Average Reward")
        if self.hanoi:
            name = "toh"
            plt.title(
                f"Towers of Hanoi Policy Iteration Average Reward for 100 Episodes Until Convergence"
            )
        else:
            name = "frozenlake"
            plt.title(
                f"Frozen Lake Policy Iteration Average Reward for 100 Episodes Until Convergence"
            )

        ax.grid(axis="x")
        file_name = f"{name}-convergence_{self.convergence}.png"
        plt.savefig("pi_results/{}".format(file_name))
        plt.close()

    def _get_hanoi_score(self):
        misses = 0
        steps_list = []
        total_rewards = []
        # for episode in range(500):
        # reset function doesn't work like we'd expect for this mdp
        self.env.s = 0
        current_state = self.env.s
        print(f"start state: {self.env.all_states[current_state]}")
        total_reward = 0
        steps = 0

        for i in range(500):
            action = self.policy[current_state][0]
            next_state, reward, done, _ = self.env.step(action)
            steps += 1
            print(
                f"step {steps} - current state: {self.env.all_states[current_state]}, done: {done} "
            )
            nstate = self.env.all_states[next_state]
            total_reward += reward
            if (
                reward > 0
                and done
                and self.env.all_states[next_state] == self.env.goal_state
            ):
                print("You have got to the end after {} steps".format(steps))
                print(
                    f"step {steps} - current state: {self.env.all_states[next_state]}"
                )
                steps_list.append(steps)
                break
            elif done:
                print("You missed")
                misses += 1
            else:
                c = self.env.all_states[current_state]
                n = self.env.all_states[next_state]
                current_state = next_state
        total_rewards.append(total_reward)
        if self.env.all_states[next_state] != self.env.goal_state:
            print(f"step {steps} - current state: {self.env.all_states[current_state]}")

        print("----------------------- Policy Iteration -----------------------")
        print(
            "You took {:.0f} steps, missed {} times, got {:.0f} reward, ended at state: {}".format(
                steps, misses, total_reward, current_state
            )
        )
        print("----------------------- Policy Iteration End -----------------------")
        print()
        print()


if __name__ == "__main__":
    # env = WindyCliffWalkingEnv()
    # pi = PolicyIteration(env, debug=False, max_iterations=10000)
    # policy = [[0] for i in range(env.nS)]
    # # pi.policy_evaluation(policy)
    # pi.policy_iteration()
    initial_state = ((6, 5, 4, 3, 2, 1, 0), (), ())
    goal_state = ((), (), (6, 5, 4, 3, 2, 1, 0))
    env = TohEnv(initial_state=initial_state, goal_state=goal_state)
    pi = PolicyIteration(env, debug=False)

"""
Evaluated 14 policies
V: [40.837316566580014, 46.48590729620001, 52.76211921800001, 59.73568802000001, 67.48409780000001, 76.09344200000001, 85.65938000000001, 96.28820000000002, 108.09800000000001, 121.22000000000001, 135.8, 152.0, 46.48590729620001, 52.76211921800001, 59.73568802000001, 67.48409780000001, 76.09344200000001, 85.65938000000001, 96.28820000000002, 108.09800000000001, 121.22000000000001, 135.8, 152.0, 170.0, 52.76211921800001, 59.73568802000001, 67.48409780000001, 76.09344200000001, 85.65938000000001, 96.28820000000002, 108.09800000000001, 121.22000000000001, 135.8, 152.0, 170.0, 190.0, 46.48590729620001, 52.76211921800001, 59.73568802000001, 67.48409780000001, 76.09344200000001, 85.65938000000001, 96.28820000000002, 108.09800000000001, 121.22000000000001, 135.8, 190.0, 100.0]
Policy: [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [2], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [2], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [1]]
"""
