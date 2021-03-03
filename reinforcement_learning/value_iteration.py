import time

import matplotlib.pyplot as plt
import numpy as np


def print_game_grid(flat_arr, length_of_board=8):
    # 0: up, 1: right, 2: down, 3: left
    directions = ["↑", "→", "↓", "←"]

    for i in range(len(flat_arr)):
        print(directions[flat_arr[i]], end=" ")
        if (i + 1) % length_of_board == 0:
            print("\n")


class ValueIteration:
    def __init__(
        self,
        env,
        max_iterations=1000,
        gamma=0.9,
        theta=1e-20,
        hanoi=False,
        plot_interval=0,
        max_steps=200,
    ):
        self.env = env
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.theta = theta
        self.state_utilities = [0 for i in range(self.env.nS)]
        self.policy = [0 for i in range(self.env.nS)]
        self.hanoi = hanoi
        self.plot_interval = plot_interval
        self.convergence = 0
        self.max_steps = max_steps

        self.time_to_convergence = None

    def _one_step_lookahead(self, state_index, v):
        """
        Helper function to calculate a state value function

        :param state_index: index of the current state to consider
        :param v: value to use as an estimator (vector of length nS)
        :return: expected value of each action for the given state (vector of length nA)
        """
        action_values = np.zeros(self.env.nA)
        for action_index in range(self.env.nA):
            action_values[action_index] += self._calculate_bellman_equation(
                action_index, state_index, v
            )
        return action_values

    def _calculate_bellman_equation(self, action_index, state_index, state_utilities):
        action_utility = 0
        # Calculate expected action utility since there is a possibility you do not go where your action states
        for probability, next_state, reward, done in self.env.P[state_index][
            action_index
        ]:
            # Result of action given the probability it would occur
            state_action_value = probability * (
                reward + self.gamma * state_utilities[next_state]
            )
            action_utility += state_action_value
        return action_utility

    def print_game_grid(self, flat_arr, length_of_board=8):
        # 0: up, 1: right, 2: down, 3: left
        directions = ["↑", "→", "↓", "←"]

        for i in range(len(flat_arr)):
            print(directions[flat_arr[i]], end=" ")
            if (i + 1) % length_of_board == 0:
                print("\n")

    def run(self):
        start_time = time.time()
        print("----------------------- Value Iteration Begin -----------------------")
        # For each iteration
        intervals = []
        avg_rewards = []
        for i in range(int(self.max_iterations)):
            delta = 0

            # For each state
            for state_index in range(self.env.nS):

                actions = self._one_step_lookahead(state_index, self.state_utilities)

                best_action_utility = max(actions)
                delta = max(
                    delta,
                    np.abs(self.state_utilities[state_index] - best_action_utility),
                )
                self.state_utilities[state_index] = best_action_utility
                # NOTE: This will stop growing since gamma is less than 1

            if delta < self.theta:
                convergence_time = time.time()
                self.time_to_convergence = convergence_time - start_time
                for state_index in range(self.env.nS):
                    action = self._one_step_lookahead(state_index, self.state_utilities)
                    best_action = np.argmax(action)
                    self.policy[state_index] = best_action
                wins, total_reward, average_reward = self.play_episodes()
                intervals.append(i)
                avg_rewards.append(average_reward)
                print(
                    f"Value-iteration converged at iteration#{i} and took {self.time_to_convergence} seconds to converge"
                )
                self.convergence = i
                break

            if self.plot_interval > 0 and i % self.plot_interval == 0:
                for state_index in range(self.env.nS):
                    action = self._one_step_lookahead(state_index, self.state_utilities)
                    best_action = np.argmax(action)
                    self.policy[state_index] = best_action
                wins, total_reward, average_reward = self.play_episodes()
                print(f"i: {i}, average reward: {average_reward}")
                intervals.append(i)
                avg_rewards.append(average_reward)

        print(f"Value-iteration converged at iteration#{i}.")

        # PLOT PERFORMANCE VS. ITERATION
        if self.plot_interval > 0:
            print("plotting value iteration")
            self._plot(intervals, avg_rewards)

    def play_episodes(self, n_episodes=100, max_steps=200):
        wins = 0
        total_reward = 0
        for episode in range(n_episodes):
            terminated = False
            state = self.env.reset()
            step = 0
            while not terminated and step < max_steps:
                # Select best action to perform in a current state
                action = self.policy[state]
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
        plt.plot(xi, y, marker=".", linestyle="--", color="b")

        plt.xlabel("Number of Iterations")
        plt.ylabel("Average Reward")
        if self.hanoi:
            name = "toh"
            plt.title(
                f"Towers of Hanoi Value Iteration Average Reward for 100 Episodes Until Convergence"
            )
        else:
            name = "frozenlake"
            plt.title(
                f"Frozen Lake Value Iteration Average Reward for 100 Episodes Until Convergence"
            )

        ax.grid(axis="x")
        file_name = f"{name}-convergence_{self.convergence}.png"
        plt.savefig("vi_results/{}".format(file_name))
        plt.close()

    def _get_score(self):
        misses = 0
        steps_list = []
        current_state = self.env.reset()
        total_reward = 0
        for i in range(500):
            steps = 0
            while True:
                action = self.policy[current_state]
                current_state, reward, done, _ = self.env.step(action)
                steps += 1
                total_reward += reward
                if done and reward > 0:
                    print("You have got to the end after {} steps".format(steps))
                    steps_list.append(steps)
                    break
                if done:
                    print("You missed")
                    misses += 1
                    break
                # elif current_state == 36:
                #     print("You fell down the cliff!")
                #     misses += 1
                #     break
        print("----------------------- Value Iteration -----------------------")
        print(
            "You took {:.0f} steps, got {} reward, ended at state: {}".format(
                np.mean(steps_list), total_reward, current_state
            )
        )
        print_game_grid(self.policy)
        print("----------------------- Value Iteration End -----------------------")
        print()

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
            action = self.policy[current_state]
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

        print("----------------------- Value Iteration -----------------------")
        print(
            "You took {:.0f} steps, missed {} times, got {:.0f} reward, ended at state: {}".format(
                steps, misses, total_reward, current_state
            )
        )
        print("----------------------- Value Iteration End -----------------------")
        print()
        print()


"""
  0: '⬆',
            1: '➡',
            2: '⬇',
            3: '⬅'
"""
if __name__ == "__main__":
    env = WindyCliffWalkingEnv()
    env.reset()
    value_iteration = ValueIteration(env, max_iterations=5000, gamma=0.9, theta=1e-23)
    value_iteration.run()
    print("-------------Optimal Utilities---------------")
    print_game_grid(value_iteration.state_utilities)

    print("-------------Optimal Policy---------------")
    print_game_grid(value_iteration.policy)
