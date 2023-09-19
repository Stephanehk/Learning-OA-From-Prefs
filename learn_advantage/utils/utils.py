import math

import numpy as np
import torch

from learn_advantage.algorithms.rl_algos import (
    value_iteration,
    build_pi,
    learn_successor_feature_iter,
)
from learn_advantage.env.generate_random_policies import calc_advantage, calc_value


def is_in_blocked_area(x, y, board):
    """
    Returns true if the inputted coordinated, (x,y), are in an inaccessible region of the boar

    Input:
    - x,y: input coordinates
    - board: the board configuration
    """
    return board[x][y] in {2, 8}


def sigmoid(val):
    """
    Sigmoid function
    """
    val = np.clip(val, -100, 100)
    return 1 / (1 + math.exp(-val))


def disp_mmv(arr, title, axis):
    """
    Prints the mean, median, and variance for an array
    """
    print("Mean " + title + ": " + str(np.mean(arr, axis=axis)))
    print("Median " + title + ": " + str(np.median(arr, axis=axis)))
    print(title + " Variance: " + str(np.var(arr, axis=axis)))


def remove_gt_succ_feat(
    *, succ_feats, succ_feats_q, action_succ_feats, gt_rew_vec, gamma, env
):
    """
    Removes the successor feature of the optimal policy under the ground truth reward function from a list of successor features

    Input:
    - succ_feats: a list of state successor features
    - succ_feats_q: a list of state, action successor features
    - action_succ_feats: a list of action successor features
    - gt_rew_vec: the ground truth reward function
    - gamma: the discount value

    Ouput:
    - mod_succ_feats: a list of state successor features without the successor feature of the optimal policy under the ground truth reward function
    - mod_succ_feats_q: a list of state action successor features without the successor feature of the optimal policy under the ground truth reward function
    - mod_action_succ_feats: a list of action successor features without the successor feature of the optimal policy under the ground truth reward function
    """
    vec = np.array(gt_rew_vec)
    _, Qs = value_iteration(rew_vec=vec, gamma=gamma, env=env)
    pi = build_pi(Qs, env=env)
    gt_succ_feat, gt_action_succ_feat, gt_q_succ_feat = learn_successor_feature_iter(
        pi, gamma, rew_vec=vec, env=env
    )
    mod_succ_feats = []
    mod_succ_feats_q = []
    mod_action_succ_feats = []

    for succ_feat, succ_feat_q, action_succ_feat in zip(
        succ_feats, succ_feats_q, action_succ_feats
    ):
        if not np.array_equal(gt_succ_feat, succ_feat):
            mod_succ_feats.append(succ_feat)
        if not np.array_equal(gt_q_succ_feat, succ_feat_q):
            mod_succ_feats_q.append(succ_feat_q)
        if not np.array_equal(gt_action_succ_feat, action_succ_feat):
            mod_action_succ_feats.append(action_succ_feat)

    return mod_succ_feats, mod_succ_feats_q, mod_action_succ_feats


def get_gt_regret(
    args,
    *,
    precomputed_vf,
    x,
    r,
    t1_ss=None,
    t1_es=None,
    t2_ss=None,
    t2_es=None,
    actions=None,
    states=None,
    gt_rew_vec=None,
    env=None,
    regret_weights=None,
):
    """
    Calculates the ground truth regret for a single segment pairs

    Input:
        - x: a list of each segment pairs features
        - r: a list of each segment pairs partial return
        - t1_ss, t1_es, ...: the start and end state for each segment
        - actions/states: a list of state action pairs for each segment, only used for stochastic MDPs
        - gt_rew_vec: the ground truth reward vector, None if using the default delivery domain
        - env: the environment, None if using the default delivery domain

    Output:
        - ground truth regret for each segment
    """

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    # If include_actions=True, we calculate regret fusign Eq. 4, which is needed if the environment has stochastic transitions.
    # Otherwise, we calculate regret using Eq.3, which is faster and requires less info to be loaded.
    if actions is not None and args.include_actions:
        seg1_actions = actions[0]
        seg2_actions = actions[1]

        seg1_states = states[0]
        seg2_states = states[1]

        assert tuple(seg1_states[0]) == tuple(t1_ss)
        assert tuple(seg1_states[-1]) == tuple(t1_es)
        assert tuple(seg2_states[0]) == tuple(t2_ss)
        assert tuple(seg2_states[-1]) == tuple(t2_es)

    if args.include_actions:
        r1_cer = calc_advantage(
            precomputed_vf, seg1_states, seg1_actions, gt_rew_vec, env
        )
        r2_cer = calc_advantage(
            precomputed_vf, seg2_states, seg2_actions, gt_rew_vec, env
        )

    else:
        x = np.array(x)

        if regret_weights is None:
            w0 = 1
            w1 = 1
            w2 = 1
        else:
            w0 = regret_weights[0]
            w1 = regret_weights[1]
            w2 = regret_weights[2]

        r1_cer = (
            w0 * r[0]
            + w1 * calc_value(precomputed_vf, t1_es, gt_rew_vec, env)
            - w2 * calc_value(precomputed_vf, t1_ss, gt_rew_vec, env)
        )
        r2_cer = (
            w0 * r[1]
            + w1 * calc_value(precomputed_vf, t2_es, gt_rew_vec, env)
            - w2 * calc_value(precomputed_vf, t2_ss, gt_rew_vec, env)
        )
    r1_cer = np.round(r1_cer, 2)
    r2_cer = np.round(r2_cer, 2)
    return r1_cer, r2_cer
