import time

import matplotlib.pyplot as plt
import numpy as np


def calculate_bellman_equation(env, action_index, state_index, state_utilities, gamma):
    action_utility = 0
    # Calculate expected action utility since there is a possibility you do not go where your action states
    for probability, next_state, reward, done in env.P[state_index][action_index]:
        # Result of action given the probability it would occur
        state_action_value = probability * (
            reward + gamma * state_utilities[next_state]
        )
        action_utility += state_action_value
    return action_utility


def q_table_to_policy(env, q_table):
    policy = [0 for i in range(env.nS)]
    for state_index in range(env.nS):
        policy[state_index] = np.argmax(q_table[state_index])
    return policy


def plot_policies(env, vi=None, pi=None, ql=None, hanoi=False):
    """
    xi = states
    y is the actions for each state given a certain algorithm
    """
    if vi:
        vi_policy = vi.policy
    if pi:
        pi_policy = [i[0] for i in pi.policy]
    if ql:
        ql_policy = q_table_to_policy(env, ql.q_table)
    states = range(env.nS)
    plt.rcParams["figure.figsize"] = (12, 6)
    fig, ax = plt.subplots()
    if hanoi:
        name = "toh_"
        title = "Towers of Hanoi"
    else:
        name = "frozenlake_"
        title = "Frozen Lake"

    if vi:
        name += "vi"
        ax.scatter(states, vi_policy, marker="*", label="Value Iteration", color="b")
    if pi:
        name += "pi"
        ax.scatter(states, pi_policy, marker="o", label="Policy Iteration", color="g")
    if ql:
        name += "ql"
        ax.scatter(states, ql_policy, marker=".", label="Q-Learning", color="r")

    title += " Policies"
    plt.title(title)
    ax.legend()
    plt.xlabel("States")
    plt.ylabel("Action")

    ax.grid(axis="x")
    file_name = f"{name}_policies_{time.time()}.png"
    plt.savefig("policy_results/{}".format(file_name))
    plt.close()
