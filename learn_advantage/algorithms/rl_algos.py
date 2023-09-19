import random

import numpy as np
import torch

from learn_advantage.env.grid_world import GridWorldEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_random_reward_vector():
    """
    Generates a random reward vector by sampling without replacement from the set [-1,50,-50,1,-1,-2]

    Output:
    - vector: The randomly generated reward vector
    """
    space = [-1, 50, -50, 1, -1, -2]
    vector = []
    for _ in range(6):
        s = random.choice(space)
        space.remove(s)
        vector.append(s)
    return vector


def learn_successor_feature_iter(pi, Fgamma=0.999, rew_vec=None, env=None):
    """
    Uses value iteration to find the state, action, and (s,a) pair successor features (SFs).

    Output:
    - psi: The state SFs
    - psi_actions: The (s,a) pair SFs
    - psi_Q: The action SFs
    """
    if env is None:
        env = GridWorldEnv("2021-07-29_sparseboard2-notrap")
    if isinstance(rew_vec, np.ndarray):
        env.set_custom_reward_function(rew_vec[0:6])

    THETA = 0.001
    # initialize Q
    num_actions = len(GridWorldEnv.actions)
    psi = [
        [np.zeros(env.feature_size) for i in range(env.width)]
        for j in range(env.height)
    ]
    psi_actions = [
        [np.zeros(num_actions * env.width * env.height) for i in range(env.width)]
        for j in range(env.height)
    ]
    psi_Q = [
        [
            [np.zeros(env.feature_size) for a in range(num_actions)]
            for i in range(env.width)
        ]
        for j in range(env.height)
    ]
    # iterativley learn state value
    while True:
        delta = 0
        new_psi = np.copy(psi)

        for i, j in env.positions():
            if env.is_blocked(i, j):
                continue
            # total = 0

            state_psi = []
            action_psi = []
            for trans in pi[(i, j)]:
                prob, a_index = trans
                next_state, _, done, phi = env.get_next_state((i, j), a_index)

                action_phi = np.zeros((env.height, env.width, num_actions))
                action_phi[i][j][a_index] = 1
                action_phi = np.ravel(action_phi)

                ni, nj = next_state
                if not done:
                    psi_sas = prob * (phi + Fgamma * psi[ni][nj])
                    psi_q = phi + Fgamma * psi[ni][nj]
                    action_feat = prob * (action_phi + Fgamma * psi_actions[ni][nj])
                else:
                    psi_sas = np.zeros(env.feature_size)
                    psi_q = np.zeros(env.feature_size)
                    action_feat = np.zeros(num_actions * env.width * env.height)

                psi_Q[i][j][a_index] = psi_q
                state_psi.append(psi_sas)
                action_psi.append(action_feat)

            psi_actions[i][j] = sum(action_psi)
            new_psi[i][j] = sum(state_psi)
            delta = max(delta, np.sum(np.abs(psi[i][j] - new_psi[i][j])))

        psi = new_psi

        if delta < THETA:
            break
    return psi, np.array(psi_actions), psi_Q


def build_reward_from_nn_feats(model, env):
    height = env.height
    width = env.width

    pred_OAF = np.zeros((height, width, len(env.actions)))

    for i, j in env.positions():
        with torch.no_grad():
            for a_i in range(len(env.actions)):
                state_embedding = torch.zeros((height, width))
                state_embedding[i][j] = 1

                action_embedding = torch.zeros(len(env.actions))
                action_embedding[a_i] = 1

                sa_embedding = torch.cat((state_embedding.flatten(), action_embedding))

                pred_OAF[i][j][a_i] = model.get_trans_val(
                    sa_embedding.to(device).float()
                ).cpu()
    return pred_OAF


def build_pi_from_nn_feats(model, env):
    """
    Given a neural network model (currently only supports instances of models.RewardFunctionPRGen), create a policy
    by acting greedily over the networks predicted rewards.

    Input:
    - model: The neural network model, must be an instance of models.RewardFunctionPRGen
    - env: The GridWorld object for the current MDP. If None, use the delivery domain instantiations MDP

    Output:
    - pi: the policy, represented as a dictionary where pi(s) = [p(a1|s), p(a2|s), ...]
    """
    pi = {}

    height = env.height
    width = env.width
    for i, j in env.positions():
        with torch.no_grad():
            weights = []
            for a_i in range(len(env.actions)):
                state_embedding = torch.zeros((height, width))
                state_embedding[i][j] = 1

                action_embedding = torch.zeros(len(env.actions))
                action_embedding[a_i] = 1

                sa_embedding = torch.cat((state_embedding.flatten(), action_embedding))
                weights.append(
                    model.get_trans_val(sa_embedding.to(device).float()).cpu()
                )

            max_weight = np.max(weights)
            count = weights.count(max_weight)
            pi[(i, j)] = [
                (1 / count if weights[a_index] == max_weight else 0, a_index)
                for a_index in range(len(env.actions))
            ]

    return pi


def build_pi_from_feats(s_a_weights, env):
    """
    Given learned (s,a) pair weights, create a policy by acting greedily over the predicted weights.

    Input:
    - s_a_weights: The (s,a) pair weghts represented as a 1d vector
    - env: The GridWorld object for the current MDP. If None, use the delivery domain instantiations MDP

    Output:
    - pi: the policy, represented as a dictionary where pi(s) = [p(a1|s), p(a2|s), ...]
    """
    pi = {}

    height = env.height
    width = env.width

    s_a_weights = s_a_weights.reshape((height, width, len(env.actions)))

    for i, j in env.positions():
        max_weight = max(s_a_weights[i][j])
        count = s_a_weights[i][j].tolist().count(max_weight)
        pi[(i, j)] = [
            (1 / count if s_a_weights[i][j][a_index] == max_weight else 0, a_index)
            for a_index in range(len(env.actions))
        ]

    return pi


def build_pi(Q, env):
    """
    Given a learned Q function, create a policy by acting greedily over the Q values.

    Input:
    - s_a_weights: The Q function
    - env: The GridWorld object for the current MDP. If None, use the delivery domain instantiations MDP


    Output:
    - pi: the policy, represented as a dictionary where pi(s) = [p(a1|s), p(a2|s), ...]
    """
    pi = {}
    num_actions = len(GridWorldEnv.actions)

    for i, j in env.positions():
        V = max(Q[i][j])
        V_count = Q[i][j].tolist().count(V)
        pi[(i, j)] = [
            (1 / V_count if Q[i][j][a_index] == V else 0, a_index)
            for a_index in range(num_actions)
        ]
    return pi


def build_random_policy(env):
    """
    Generated a policy that uniform-randomly selects actions at each state.

    Input:
    - env: The GridWorld object for the current MDP. If None, use the delivery domain instantiations MDP

    Output:
    - pi: the policy, represented as a dictionary where pi(s) = [p(a1|s), p(a2|s), ...]
    """
    pi = {}
    num_actions = len(GridWorldEnv.actions)

    for i, j in env.positions():
        pi[(i, j)] = [
            (1.0 / float(num_actions), a_index) for a_index in range(num_actions)
        ]
    return pi


def iterative_policy_evaluation(
    pi, *, rew_vec=None, set_rand_rew=False, gamma=0.999, env=None
):
    """
    Performs iterative policy evaluation.

    Input:
    - pi: The policy, represented as a dictionary whenre pi(s) = [p(a1|s), p(a2|s), ...]
    - rew_vec: The reward vector to evaluate the policy with. If none, the default reward vector is used.
    - set_rand_rew: If true, evaluate pi with a random reward vector
    - gamma: The discount factor
    - env: The GridWorld object for the current MDP. If None, use the delivery domain instantiations MDP

    Output:
    - V: the value function
    """
    if env is None:
        env = GridWorldEnv("2021-07-29_sparseboard2-notrap")

    if isinstance(rew_vec, np.ndarray):
        env.set_custom_reward_function(rew_vec[0:6])
    elif set_rand_rew:
        env.set_custom_reward_function(rew_vec)

    THETA = 0.001
    V = np.zeros((env.height, env.width))

    # iterativley learn state value
    while True:
        delta = 0
        new_V = V.copy()
        for i, j in env.positions():
            if env.is_blocked(i, j):
                continue
            # total = 0
            state_qs = []
            for trans in pi[(i, j)]:
                prob, a_index = trans
                next_state, reward, done, _ = env.get_next_state((i, j), a_index)
                ni, nj = next_state
                if not done:
                    state_qs.append(prob * (reward + gamma * V[ni][nj]))
                else:
                    state_qs.append(prob * reward)
            new_V[i][j] = sum(state_qs)
            delta = max(delta, np.abs(V[i][j] - new_V[i][j]))

        V = new_V
        if delta < THETA:
            break

    return V


def value_iteration(
    rew_vec=None,
    set_rand_rew=False,
    gamma=0.999,
    env=None,
    is_set=False,
    extended_SF=False,
):
    """
    Performs value iteration.

    Input:
    - pi: The policy, represented as a dictionary whenre pi(s) = [p(a1|s), p(a2|s), ...]
    - rew_vec: The reward vector to evaluate the policy with. If none, the default reward vector is used.
    - set_rand_rew: If true, evaluate pi with a random reward vector
    - gamma: The discount factor
    - env: The GridWorld object for the current MDP. If None, use the delivery domain instantiations MDP
    - extended_SF: If false, treats rew_vec as a reward function. Otherwise treats rew_vec as an optimal advantage function.

    Output:
    - V: the value function
    - Qs: the Q function
    """
    if env is None:
        env = GridWorldEnv("2021-07-29_sparseboard2-notrap")

    if isinstance(rew_vec, np.ndarray) and not is_set and not extended_SF:
        env.set_custom_reward_function(rew_vec[0:6])
    elif set_rand_rew and not extended_SF:
        rand_rew_vec = get_random_reward_vector()
        env.set_custom_reward_function(rand_rew_vec)

    THETA = 0.001
    # initialize Q
    V = np.zeros((env.height, env.width))
    Qs = [
        [np.zeros(len(env.actions)) for i in range(env.width)]
        for j in range(env.height)
    ]

    num_actions = len(GridWorldEnv.actions)

    # iterativley learn state value
    while True:
        delta = 0
        new_V = V.copy()
        for i, j in env.positions():
            if env.is_blocked(i, j):
                continue
            v = V[i][j]
            for a_index in range(num_actions):
                next_state, reward, done, _ = env.get_next_state((i, j), a_index)

                if extended_SF:
                    reward = rew_vec[i][j][a_index]

                ni, nj = next_state
                if not done:
                    Q = reward + gamma * V[ni][nj]
                else:
                    Q = reward
                Qs[i][j][a_index] = Q

            new_V[i][j] = max(Qs[i][j])
            delta = max(delta, np.abs(v - new_V[i][j]))

        V = new_V
        if delta < THETA:
            break

    return V, Qs


def get_gt_avg_return(gamma=0.999, gt_rew_vec=None, env=None):
    """
    Gets the average return of the maximum entropy policy

    Input:
    - gamma: The discount factor
    - gt_rew_vec: The reward vector to evaluate the policy with. If none, the default reward vector is used.
    - env: The GridWorld object for the current MDP. If None, use the delivery domain instantiations MDP

    Output:
    - gt_avg_return: The average return of the maximum entropy policy with respect to gt_rew_vec
    """
    _, Q = value_iteration(rew_vec=gt_rew_vec, env=env, gamma=gamma)
    pi = build_pi(Q, env=env)
    V_under_gt = iterative_policy_evaluation(pi, rew_vec=gt_rew_vec, env=env)
    if env is None:
        n_starts = 92
    else:
        n_starts = env.n_starts
    gt_avg_return = np.sum(V_under_gt / n_starts)

    return gt_avg_return


def get_start_state(env):
    i = np.random.randint(0, env.height)
    j = np.random.randint(0, env.width)
    while env.is_terminal(i, j):
        i = np.random.randint(0, env.height)
        j = np.random.randint(0, env.width)
    return (i, j)


def eps_greedy(epsilon, decay_rate, current_action):
    # Choose a random number between 0 and 1
    if np.random.random(1)[0] < epsilon:
        # Exploration: Choose a random action
        chosen_action = np.random.randint(4)
    else:
        # Exploitation: Choose the current best action
        chosen_action = current_action

    # Update epsilon
    epsilon *= decay_rate

    return chosen_action, epsilon


def q_learning(
    *,
    rew_vec=None,
    gamma=0.999,
    alpha=0.1,
    env=None,
    extended_SF=False,
    n_episodes=1000,
    return_training_curve=False,
    gt_rew_vec=None,
    checkpoint_every=20,
    epsilon=0.4,
):
    if env is None:
        env = GridWorldEnv("2021-07-29_sparseboard2-notrap")

    if isinstance(rew_vec, np.ndarray) and not extended_SF:
        env.set_custom_reward_function(rew_vec[0:6])

    Qs = [
        [np.zeros(len(env.actions)) for i in range(env.width)]
        for j in range(env.height)
    ]
    training_avg_returns = []

    for episode in range(n_episodes + 1):  # +1 so that we plot the last epoch
        done = False
        state = get_start_state(env)
        n_steps = 0
        while not done and n_steps < 1000:
            n_steps += 1
            a_index = np.argmax(Qs[state[0]][state[1]])

            a_index, epsilon = eps_greedy(epsilon, 0.99, a_index)
            next_state, reward, done, _ = env.get_next_state(state, a_index)
            if extended_SF:
                reward = rew_vec[state[0]][state[1]][a_index]

            Qs[state[0]][state[1]][a_index] += alpha * (
                reward
                + gamma * np.max(Qs[next_state[0]][next_state[1]])
                - Qs[state[0]][state[1]][a_index]
            )
            state = next_state
        if episode % checkpoint_every == 0 and return_training_curve:
            assert isinstance(rew_vec, np.ndarray)
            pi = build_pi(Qs, env)
            V_under_gt = iterative_policy_evaluation(pi, rew_vec=gt_rew_vec, env=env)
            avg_return = np.sum(V_under_gt) / env.n_starts
            training_avg_returns.append(avg_return)
    if not return_training_curve:
        pi = build_pi(Qs, env)
        V_under_gt = iterative_policy_evaluation(pi, rew_vec=gt_rew_vec, env=env)
        avg_return = np.sum(V_under_gt) / env.n_starts
        return Qs, avg_return
    return Qs, training_avg_returns
