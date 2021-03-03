import time

import matplotlib.pyplot as plt
import numpy as np


class QLearning:
    def __init__(
        self,
        env,
        max_episodes=1000,
        min_episodes=0,
        learning_rate=0.8,
        max_steps=100,
        gamma=0.9,
        epsilon=1.0,
        min_epsilon=0.01,
        max_epsilon=1,
        decay_rate=0.005,
        seed=1,
        rolling_window_size=500,
        threshold=0,
        debug=False,
        hanoi=False,
        plot_interval=0,
        plot_episodes=100,
        plot_threshold=1000,
    ):
        """

        env ~ Discrete Env to run algorithm on
        episodes ~ Number of times to run algorithm on env
        learning_rate ~ rate at which to learn from each transition
        max_steps ~ Maximum amount of steps to take
        gamma ~ discount rate
        epsilon ~ Determines whether or not to take greedy action or random action. It is
        decayed by the decay rate over iterations
        min_epsilon ~ Minimum value for epsilon to decay to
        decay_rate ~ exponential decaying rate for epsilon

        NOTE: If you alter episodes make sure to change the decay rate (or other epsilons) to match it

        """
        self.env = env
        self.max_episodes = max_episodes
        self.min_episodes = min_episodes
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.decay_rate = decay_rate

        self.action_size = env.action_space.n
        self.state_size = env.nS
        self.q_table = np.zeros((self.state_size, self.action_size))
        self.episode_rewards = []
        self.seed = seed

        self.rolling_window_size = rolling_window_size
        self.threshold = threshold

        self.debug = debug

        self.hanoi = hanoi
        self.plot_interval = plot_interval
        self.plot_episodes = plot_episodes
        self.plot_threshold = plot_threshold
        self.convergence = 0
        self.time_to_converge = None

    def _find_greediest_action(self, state):
        # find action that maximizes Q value
        action = np.argmax(self.q_table[state, :])
        return action

    def _calculate_q_value(self, action, new_state, q_table, reward, state):
        new_state_optimal_action = np.argmax(q_table[new_state, :])
        # utility = r + gamma *  q(s', a') where a' is the optimal action at s'
        utility = reward + self.gamma * q_table[new_state, new_state_optimal_action]
        # Apply learning rate so: q(s, a) = q(s, a) + learning_rate * (utility - q(s, a))
        learned_q_value = q_table[state, action] + self.learning_rate * (
            utility - q_table[state, action]
        )
        return learned_q_value

    def run(self):
        print("----------------------- Q Learning Begin -----------------------")
        epsilon = self.epsilon
        wins = 0
        max_stepped_out = 0
        np.random.seed(self.seed)
        rolling_mean = RollingMean(self.rolling_window_size)
        intervals = []
        avg_rewards = []
        plot_rewards = []
        start_time = time.time()
        for episode in range(self.max_episodes):
            self.env.seed(self.seed)
            state = self.env.reset()
            episode_reward = 0

            for step in range(self.max_steps):
                random_value = np.random.random()
                # If epsilon is really high, it'll take random step more often
                if epsilon > random_value:
                    # random step
                    action = self.env.action_space.sample()
                else:
                    # greedy step
                    action = self._find_greediest_action(state)

                new_state, state_reward, done, info = self.env.step(action)
                self.q_table[state, action] = self._calculate_q_value(
                    action, new_state, self.q_table, state_reward, state
                )
                episode_reward += state_reward

                if done:
                    if state_reward > 0:
                        wins += 1
                        # print(f"WON game at state: {new_state}, with reward: {episode_reward}, for episode: {episode}, steps: {step+1}, epsilon: {epsilon}")
                    else:
                        pass
                    break

                # Start over
                state = new_state

            if step == self.max_steps - 1:
                max_stepped_out += 1

            rolling_mean.add(episode_reward)
            self.episode_rewards.append(episode_reward)

            # Formula taken from
            # https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/FrozenLake/Q%20Learning%20with%20FrozenLake.ipynb
            # Decreases epsilon slowly to 0.
            epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
                -self.decay_rate * episode
            )

            if episode % 1000 == 0 and self.debug:
                # {window_stats.mean().iloc[[-1]]
                print(
                    f"Episode: {episode}, Average Reward last 100: {rolling_mean.mean()} Epsilon value: {epsilon}, Wins so far: {wins}, Steps: {step}, State: {new_state}, times reached max steps: {max_stepped_out}"
                )

            if rolling_mean.mean() > self.threshold and episode > self.min_episodes:
                self.convergence = episode
                self.time_to_converge = time.time() - start_time
                print(
                    f"reached convergence at episode {episode} and took {self.time_to_converge} seconds"
                )
                total_reward, average_reward = self.play(self.plot_episodes)
                print(f"i: {episode}, average reward: {average_reward}")
                intervals.append(episode)
                plot_rewards.append(episode_reward)
                avg_rewards.append(average_reward)
                break

            if (
                self.plot_interval > 0
                and episode > self.plot_threshold
                and episode % self.plot_interval == 0
            ):
                plot_rewards.append(episode_reward)
                intervals.append(episode)
            # if episode > self.plot_threshold and self.plot_interval > 0 and episode % self.plot_interval == 0:
            #     total_reward, average_reward = self.play(self.plot_episodes)
            #     print(f"i: {episode}, average reward: {average_reward}")
            #     intervals.append(episode)
            #     avg_rewards.append(average_reward)

        ## PLOT PERFORMANCE VS. ITERATION
        if self.plot_interval > 0:
            print("plotting value iteration")
            self._plot(intervals, plot_rewards)

        print(f"Final Episode value: {episode}")
        print(f"Final epsilon value: {epsilon}")
        print("Final Q Table Values")
        print(self.q_table)

        print(
            "Average Reward per episode: {}".format(
                sum(self.episode_rewards) / self.max_episodes
            )
        )
        print("\n\n\n\n\n\n\n\n")

    def play(self, n_episodes=5):
        self.env.reset()
        total_reward = 0
        for episode in range(n_episodes):
            state = self.env.reset()
            step = 0
            done = False
            episode_reward = 0
            finished = False
            # print("****************************************************")
            # print("EPISODE ", episode)

            for step in range(self.max_steps):

                # Take the action (index) that have the maximum expected future reward given that state
                action = np.argmax(self.q_table[state, :])

                new_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                total_reward += reward

                if done:
                    if self.debug:
                        print("Finished With:")
                        # self.env.render()
                        print(
                            f"Number of steps: {step + 1}, Episode reward: {episode_reward}"
                        )
                    finished = True
                    break
                state = new_state

            if not finished:
                if self.debug:
                    print("Incomplete: ")
                    print(
                        f"Number of steps: {step + 1}, Episode reward: {episode_reward}"
                    )
        average_reward = total_reward / n_episodes
        return total_reward, average_reward

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
                f"Towers of Hanoi Q-Learning Average Reward for 100 Episodes Until Convergence"
            )
        else:
            name = "frozenlake"
            plt.title(
                f"Frozen Lake Q-Learning Average Reward for 100 Episodes Until Convergence"
            )

        ax.grid(axis="x")
        file_name = f"{name}-convergence_{self.convergence}.png"
        plt.savefig("ql_results/{}".format(file_name))
        plt.close()


class RollingMean:
    def __init__(self, size=100):
        self.size = size
        self.vals = []

    def add(self, val):
        self.vals.append(val)

    def mean(self):
        return sum(self.vals[-self.size :]) / self.size
