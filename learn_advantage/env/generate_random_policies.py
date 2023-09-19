import random
import os

import numpy as np

from learn_advantage.algorithms.rl_algos import (
    learn_successor_feature_iter,
    value_iteration,
    build_pi,
)


id_ = random.randrange(1000000)

if not os.path.exists("data/temp_data"):
    # This directory will be used to store data that we need during training, but which can be deleted after
    os.makedirs("data/temp_data")

if not os.path.exists("data/temp_data/saved_value_funcs"):
    os.makedirs("data/temp_data/saved_value_funcs")
if not os.path.exists("data/temp_data/saved_Q_funcs"):
    os.makedirs("data/temp_data/saved_Q_funcs")
if not os.path.exists("data/temp_data/saved_succ_feats"):
    os.makedirs("data/temp_data/saved_succ_feats")


def get_random_reward_vector(gt_rew_vec):
    """
    Compute random reward vector
    """
    if gt_rew_vec is None:
        space = [-1, 50, -50, 1, -1, -2]
    else:
        space = [50, -50, 1, -1, -2, 0, -10, 10, 5]
    vector = []
    for _ in range(6):
        s = random.choice(space)
        vector.append(s)
    return np.array(vector)


def generate_random_policy(gamma, env=None, gt_rew_vec=None):
    """
    Generate a policy for a random reward vector
    """
    vec = get_random_reward_vector(gt_rew_vec)
    _, Qs = value_iteration(rew_vec=vec, gamma=gamma, env=env)
    pi = build_pi(Qs, env=env)
    succ_feat, sa_succ_feat, gt_q_succ_feat = learn_successor_feature_iter(
        pi, gamma, rew_vec=vec, env=env
    )

    return succ_feat, sa_succ_feat, pi, gt_q_succ_feat


def is_arr_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)


def generate_all_policies(n_policies, gamma, env=None, gt_rew_vec=None):
    """
    Generate a list of policies from random reward vectors
    """
    succ_feats = []
    gt_q_succ_feats = []
    sa_succ_feats = []
    pis = []
    i = 0
    n_duplicates = (
        0  # makes sure we do not try and generate more unique policies than exist
    )
    while i < n_policies and n_duplicates < 100:
        i += 1
        succ_feat, sa_succ_feat, pi, gt_q_succ_feat = generate_random_policy(
            gamma, env, gt_rew_vec
        )
        if is_arr_in_list(succ_feat, succ_feats):
            i -= 1
            n_duplicates += 1
        else:
            succ_feats.append(succ_feat)
            gt_q_succ_feats.append(gt_q_succ_feat)
            sa_succ_feats.append(sa_succ_feat)
            pis.append(pi)
    return succ_feats, sa_succ_feats, pis, gt_q_succ_feats


def calc_advantage(precomputed_vf, states, actions, gt_rew_vec=None, env=None):
    """
    Calculate advantage under V* and Q*
    """

    if not precomputed_vf:
        V, Qs = value_iteration(rew_vec=np.array(gt_rew_vec), gamma=0.999, env=env)
        pi = build_pi(Qs, env=env)
        gt_succ_feat, _, _ = learn_successor_feature_iter(
            pi, 0.999, rew_vec=gt_rew_vec, env=env
        )
        np.save(f"data/temp_data/saved_succ_feats/gt_succ_feat_{id_}.npy", gt_succ_feat)
        np.save(f"data/temp_data/saved_value_funcs/V_{id_}.npy", V)
        np.save(f"data/temp_data/saved_Q_funcs/Qs_{id_}.npy", Qs)

    else:
        V = np.load(f"data/temp_data/saved_value_funcs/V_{id_}.npy")
        Qs = np.load(f"data/temp_data/saved_Q_funcs/Qs_{id_}.npy")

    advantage = 0
    for state, action in zip(states, actions):
        x, y = state
        advantage += V[x][y] - Qs[x][y][action]
    return -advantage


def calc_value(precomputed_vf, state, gt_rew_vec=None, env=None):
    """
    Calculate value under V*
    """

    w = gt_rew_vec
    x, y = state
    if precomputed_vf:
        gt_succ_feat = np.load(
            f"data/temp_data/saved_succ_feats/gt_succ_feat_{id_}.npy"
        )
        return np.dot(gt_succ_feat[x][y], w)

    V, Qs = value_iteration(rew_vec=np.array(gt_rew_vec), gamma=0.999, env=env)
    pi = build_pi(Qs, env=env)
    gt_succ_feat, _, _ = learn_successor_feature_iter(
        pi, 0.999, rew_vec=gt_rew_vec, env=env
    )
    np.save(f"data/temp_data/saved_succ_feats/gt_succ_feat_{id_}.npy", gt_succ_feat)
    np.save(f"data/temp_data/saved_value_funcs/V_{id_}.npy", V)
    np.save(f"data/temp_data/saved_Q_funcs/Qs_{id_}.npy", Qs)
    return np.dot(gt_succ_feat[x][y], w)
