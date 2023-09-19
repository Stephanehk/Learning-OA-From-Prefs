"""
This file runs the expirement used to produce Figure 5, testing the hypothtesis that the maximum partial return by
across all loops in an MDP when using A*_hat as the reward function determines the direction of performance between the
r_hat = A*_hat model and the Greedy A*_hat model.
"""

import pickle
import os
import gzip

import numpy as np
import matplotlib.pyplot as plt

from learn_advantage.utils.argparse_utils import parse_args, ArgsType
from learn_advantage.algorithms.rl_algos import (
    value_iteration,
    build_pi,
    build_random_policy,
    build_pi_from_feats,
    iterative_policy_evaluation,
)

args = parse_args(ArgsType.FIG_5)


def get_loops(start_x, stary_y, oaf, env):
    reverse_action_mapping = {0: 1, 1: 0, 2: 3, 3: 2}

    loop_returns = []
    for action_i in range(4):
        loop_return = 0
        next_state, _, _, _ = env.get_next_state((start_x, stary_y), action_i)
        loop_return += oaf[start_x][stary_y][action_i]

        if env.is_terminal(next_state[0], next_state[1]):
            continue
        # Whatever action we initially took ended up back in the same state
        if next_state == (start_x, stary_y):
            loop_returns.append(loop_return)
            continue

        next_action_i = reverse_action_mapping[action_i]
        next_next_state, _, _, _ = env.get_next_state(next_state, next_action_i)
        loop_return += oaf[next_state[0]][next_state[1]][next_action_i]
        loop_returns.append(loop_return)
        # make sure we actually did a loop (ie: went back to state (x,y) after 2 actions)

        assert next_next_state == (start_x, stary_y)
    return loop_returns


def main():
    all_num_prefs = [int(item) for item in args.num_prefs.split(",")]
    modes = [
        "sigmoid" if item == "stochastic" else item for item in args.modes.split(",")
    ]

    start_trial = args.start_MDP
    end_trial = args.end_MDP
    MDP_dir = args.MDP_dir
    dir_name = args.output_dir_prefix
    all_extra_details = list(args.all_extra_details.split(","))

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({"font.size": 24})
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["lines.markersize"] = 4
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    # Get the value function of 1) a uniformly random policy 2) an optimal policy
    gt_Vs = []
    random_Vs = []
    for trial in range(start_trial, end_trial):
        gt_rew_vec = np.load(
            gzip.GzipFile(MDP_dir + "MDP_" + str(trial) + "gt_rew_vec.npy.gz", "r")
        )

        with open(MDP_dir + "MDP_" + str(trial) + "env.pickle", "rb") as rf:
            env = pickle.load(rf)
            if trial < 300:
                env.generate_transition_probs()
                env.set_custom_reward_function(gt_rew_vec)

        gt_V, _ = value_iteration(rew_vec=gt_rew_vec, env=env)
        gt_Vs.append(gt_V)

        random_pi = build_random_policy(env=env)
        V_under_random_pi = iterative_policy_evaluation(
            random_pi, rew_vec=np.array(gt_rew_vec), env=env
        )
        random_Vs.append(V_under_random_pi)

    max_loop_prs = []
    mean_scaled_ret_diff = []

    color_1_hex = "#00539C"
    color_2_hex = "#EEA47F"

    color_1_rgba = (0, 0.325, 0.612, 0.4)
    color_2_rgba = (0.933, 0.643, 0.498, 0.4)

    mdp2color = []
    mdp2face_color = []
    red_pos_x_no_ties = 0
    red_pos_x_oaf_better = 0
    red_neg_x_no_ties = 0
    red_neg_x_oaf_better = 0

    blue_pos_x_no_ties = 0
    blue_pos_x_oaf_better = 0
    blue_neg_x_no_ties = 0
    blue_neg_x_oaf_better = 0

    for extra_details in all_extra_details:
        for mode in modes:
            for num_prefs in all_num_prefs:
                for trial in range(start_trial, end_trial):
                    gt_rew_vec = np.load(
                        gzip.GzipFile(
                            MDP_dir + "MDP_" + str(trial) + "gt_rew_vec.npy.gz", "r"
                        )
                    )

                    with open(MDP_dir + "MDP_" + str(trial) + "env.pickle", "rb") as rf:
                        env = pickle.load(rf)
                        if trial < 300:
                            env.generate_transition_probs()
                            env.set_custom_reward_function(gt_rew_vec)

                    file_name = "_".join(
                        [
                            f"{extra_details}",
                            f"{trial}",
                            "True_regret_pr",
                            f"mode={mode}",
                            "extended_SF=True",
                            "generalize_SF=False",
                            f"num_prefs={num_prefs}",
                            "rew_vects.npy",
                        ]
                    )

                    path = "/".join([dir_name + "checkpointed_reward_vecs", file_name])
                    oaf = np.load(path)
                    oaf = oaf[-1].reshape((env.height, env.width, 4))

                    oaf_pi = build_pi_from_feats(oaf, env)

                    # Treat A*_hat as a reward function and get the induced policy
                    rew_vect = oaf.reshape((env.height, env.width, 4))
                    _, pr_Q = value_iteration(
                        rew_vec=rew_vect, env=env, extended_SF=True
                    )
                    pr_pi = build_pi(np.array(pr_Q), env=env)

                    # Get the average returns of the Greedy A*_hat policy and the r_hat=A*_hat policy under the true reward function
                    oaf_V_under_gt = iterative_policy_evaluation(
                        oaf_pi, rew_vec=np.array(gt_rew_vec), env=env
                    )
                    pr_V_under_gt = iterative_policy_evaluation(
                        pr_pi, rew_vec=np.array(gt_rew_vec), env=env
                    )

                    # Stores the return (under r=A*_hat) of each possible loop of size less than 2 in each MDP.
                    mdp_loop_rets = []
                    for i in range(env.height):
                        for j in range(env.width):
                            if env.is_terminal(i, j):
                                continue

                            # Get the return (under r=A*_hat) of each loop starting from (i,j) of size <= 2 transitions in the MDP.
                            out_loop_returns = get_loops(i, j, oaf, env)
                            mdp_loop_rets.extend(out_loop_returns)

                            # In all MDPs >= 120, it is optimal to loop forever
                            if trial >= 120:
                                color = color_2_hex
                                face_color = color_2_rgba
                            # In all MDPs < 120, it is optimal to eventually terminate
                            else:
                                color = color_1_hex
                                face_color = color_1_rgba

                    max_loop_prs.append(np.max(mdp_loop_rets))
                    mdp2color.append(color)
                    mdp2face_color.append(face_color)

                    # Compute the scaled returns of the Greedy A*_hat policy and the r_hat=A*_hat policy from their average returns.
                    oaf_mean_scaled_ret = (
                        (np.sum(oaf_V_under_gt) / env.n_starts)
                        - (np.sum(random_Vs[trial - start_trial]) / env.n_starts)
                    ) / (
                        (np.sum(gt_Vs[trial - start_trial]) / env.n_starts)
                        - (np.sum(random_Vs[trial - start_trial]) / env.n_starts)
                    )
                    pr_mean_scaled_ret = (
                        (np.sum(pr_V_under_gt) / env.n_starts)
                        - (np.sum(random_Vs[trial - start_trial]) / env.n_starts)
                    ) / (
                        (np.sum(gt_Vs[trial - start_trial]) / env.n_starts)
                        - (np.sum(random_Vs[trial - start_trial]) / env.n_starts)
                    )
                    # Clip scaled returns
                    oaf_mean_scaled_ret = np.clip(oaf_mean_scaled_ret, -1, 1)
                    pr_mean_scaled_ret = np.clip(pr_mean_scaled_ret, -1, 1)

                    mean_scaled_ret_diff.append(
                        pr_mean_scaled_ret - oaf_mean_scaled_ret
                    )

                    if trial < 160:
                        if np.max(mdp_loop_rets) > 0:
                            # if pr_mean_scaled_ret != oaf_mean_scaled_ret:
                            if pr_mean_scaled_ret != oaf_mean_scaled_ret:
                                blue_pos_x_no_ties += 1
                            if oaf_mean_scaled_ret > pr_mean_scaled_ret:
                                blue_pos_x_oaf_better += 1
                        else:
                            if pr_mean_scaled_ret != oaf_mean_scaled_ret:
                                blue_neg_x_no_ties += 1
                            if oaf_mean_scaled_ret > pr_mean_scaled_ret:
                                blue_neg_x_oaf_better += 1
                    elif np.max(mdp_loop_rets) > 0:
                        if pr_mean_scaled_ret != oaf_mean_scaled_ret:
                            red_pos_x_no_ties += 1
                        if oaf_mean_scaled_ret > pr_mean_scaled_ret:
                            red_pos_x_oaf_better += 1
                    else:
                        if pr_mean_scaled_ret != oaf_mean_scaled_ret:
                            red_neg_x_no_ties += 1
                        if oaf_mean_scaled_ret > pr_mean_scaled_ret:
                            red_neg_x_oaf_better += 1

    fig, ax = plt.subplots()
    plt.scatter(
        max_loop_prs,
        mean_scaled_ret_diff,
        marker="o",
        edgecolor=mdp2color,
        facecolor=mdp2face_color,
    )

    xmin, xmax = ax.get_xlim()
    y_min = -2.1
    y_max = 2.1

    plt.axvline(x=0, ymin=-2, ymax=2, color="grey", linestyle="--")
    plt.ylim(y_min, y_max)
    plt.xlim(xmin, xmax)
    plt.yticks([-2, -1, 0, 1, 2])
    ax.fill_between([xmin, 0], y_min, 0, alpha=0.15, color=color_2_hex)  # red
    ax.fill_between([0, xmax], 0, y_max, alpha=0.15, color=color_2_hex)  # red

    ax.fill_between([xmin, 0], 0, y_max, alpha=0.15, color=color_1_hex)  # red
    ax.fill_between([0, xmax], y_min, 0, alpha=0.15, color=color_1_hex)  # red

    n_correct = 0
    n_total = 0
    for x, y, c in zip(max_loop_prs, mean_scaled_ret_diff, mdp2color):
        if abs(y) < 0.1:
            continue
        if c == "#00539C":  # blue
            if (x < 0 and y > 0) or (x > 0 and y < 0):
                n_correct += 1
            n_total += 1

        elif c == "#EEA47F":  # red
            if (x < 0 and y < 0) or (x > 0 and y > 0):
                n_correct += 1
            n_total += 1
        else:
            assert False

    print(f"Number of points that matched our hypothesis: {n_correct}/{n_total}")

    plt.xlabel("Maximum loop return")
    plt.tight_layout()

    mode_ = ""
    if "deterministic" in modes:
        mode_ += "_deterministic_prefs_"
    if "sigmoid" in modes:
        mode_ += "_stochastic_prefs_"

    extra_details_ = ""
    if "two_trans" in all_extra_details:
        extra_details_ += "_segment_length=2"
    if "single_trans" in all_extra_details:
        extra_details_ += "_segment_length=1"
    fig.set_size_inches(8, 5)

    if not os.path.exists("data/results/AAAI_loop_plots"):
        os.makedirs("data/results/AAAI_loop_plots")

    fig_file_name = f"data/results/AAAI_loop_plots/{all_num_prefs}{mode_}{(start_trial, end_trial)}{extra_details_}"
    plt.savefig(fig_file_name, dpi=300)
    plt.clf()


if __name__ == "__main__":
    main()
