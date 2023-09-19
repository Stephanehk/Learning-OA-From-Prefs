"""
This file runs the expirement used to produce Figure 6, testing the assumption that using the learned optimal
advantage function as a reward function results in a highly shaped reward. A set of learned A*_hat functions
is read in, and we then compare Q learning when the reward function is set to A*_hat versus using the true reward
function.

A Wilcoxon paired signed rank test is also performed on the areas above each Q learning training curve.
"""
import pickle
import gzip

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from learn_advantage.utils.argparse_utils import parse_args, ArgsType
from learn_advantage.algorithms.rl_algos import (
    build_random_policy,
    get_gt_avg_return,
    iterative_policy_evaluation,
    build_pi_from_feats,
    q_learning,
)

args = parse_args(ArgsType.FIG_6)


def main():
    preference_model = args.preference_model  # how we generate prefs
    preference_assum = args.preference_assum  # how we learn prefs
    model = preference_model + "_" + preference_assum

    num_prefs = args.num_prefs
    mode = args.mode
    if mode == "stochastic":
        mode = "sigmoid"

    start_MDP = args.start_MDP
    end_MDP = args.end_MDP
    MDP_dir = args.MDP_dir
    dir_name = args.output_dir_prefix
    extra_details = args.extra_details

    n_updates_plot = args.n_episodes
    checkpoint_every = args.checkpoint_freq
    LR = args.LR

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({"font.size": 24})
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    plot_interval = int(n_updates_plot / checkpoint_every) + 1

    r_a_hat_optimals = np.zeros(plot_interval)
    r_r_true_optimals = np.zeros(plot_interval)

    r_a_hat_scaled_rets = np.zeros(plot_interval)
    r_r_true_scaled_rets = np.zeros(plot_interval)

    r_a_hat_scaled_rets_raw = [[] for _ in range((end_MDP - start_MDP))]
    r_r_true_scaled_rets_raw = [[] for _ in range(end_MDP - start_MDP)]

    oaf_scaled_rets = np.zeros(plot_interval)

    print(
        f"Q learning policies for MDPs {start_MDP} to {end_MDP} with advantage approximates learned from {num_prefs} preferences..."
    )

    for trial in range(start_MDP, end_MDP):
        # Load the MDP
        gt_rew_vec = np.load(
            gzip.GzipFile(MDP_dir + "MDP_" + str(trial) + "gt_rew_vec.npy.gz", "r")
        )
        with open(MDP_dir + "MDP_" + str(trial) + "env.pickle", "rb") as rf:
            env = pickle.load(rf)
            if trial < 300:
                env.generate_transition_probs()
                env.set_custom_reward_function(gt_rew_vec)

        # Load the learned A*_hat(s,a) function
        file_name = "_".join(
            [
                f"{extra_details}",
                f"{trial}",
                "True",
                f"{model}",
                f"mode={mode}",
                "extended_SF=True",
                "generalize_SF=False",
                f"num_prefs={num_prefs}",
                "rew_vects.npy",
            ]
        )

        path = "/".join([dir_name + "checkpointed_reward_vecs", file_name])
        # Note: right now we are just taking the last A*_hat(s,a) function that was checkpointed
        oaf = np.load(path)[-1]

        # compute the avg. return for a unfiromly random policy and for an optimal policy
        gt_avg_return = get_gt_avg_return(gt_rew_vec=gt_rew_vec, env=env)
        random_pi = build_random_policy(env=env)
        V_under_random_pi = iterative_policy_evaluation(
            random_pi, rew_vec=np.array(gt_rew_vec), env=env
        )
        random_avg_return = np.sum(V_under_random_pi) / env.n_starts

        # compute the scaled return of the Greedy A*_hat policy
        oaf_pi = build_pi_from_feats(oaf, env)
        oaf_V_under_gt = iterative_policy_evaluation(
            oaf_pi, rew_vec=np.array(gt_rew_vec), env=env
        )
        oaf_avg_return = np.sum(oaf_V_under_gt) / env.n_starts
        oaf_scaled_return = (oaf_avg_return - random_avg_return) / (
            gt_avg_return - random_avg_return
        )

        oaf = oaf.reshape((env.height, env.width, 4))

        # Performing Q Learning, with r_hat=A*_hat
        _, r_a_hat_avg_returns_q_learning = q_learning(
            rew_vec=oaf,
            alpha=LR,
            env=env,
            extended_SF=True,
            n_episodes=n_updates_plot,
            return_training_curve=True,
            gt_rew_vec=gt_rew_vec,
            checkpoint_every=checkpoint_every,
        )

        # Performing Q Learning, with r_hat = r
        _, r_r_true_avg_returns_q_learning = q_learning(
            rew_vec=gt_rew_vec,
            alpha=LR,
            env=env,
            extended_SF=False,
            n_episodes=n_updates_plot,
            return_training_curve=True,
            gt_rew_vec=gt_rew_vec,
            checkpoint_every=checkpoint_every,
        )

        # Compute the scaled returns given the average returns
        for i in range(plot_interval):
            r_a_hat_q_scaled_return = (
                r_a_hat_avg_returns_q_learning[i] - random_avg_return
            ) / (gt_avg_return - random_avg_return)
            r_r_true_q_scaled_return = (
                r_r_true_avg_returns_q_learning[i] - random_avg_return
            ) / (gt_avg_return - random_avg_return)

            if r_a_hat_q_scaled_return >= 0.9:
                r_a_hat_optimals[i] += 1
            if r_r_true_q_scaled_return >= 0.9:
                r_r_true_optimals[i] += 1

            r_a_hat_scaled_rets[i] += np.clip(r_a_hat_q_scaled_return, -1, 1)
            r_r_true_scaled_rets[i] += np.clip(r_r_true_q_scaled_return, -1, 1)
            oaf_scaled_rets[i] += np.clip(oaf_scaled_return, -1, 1)

            r_a_hat_scaled_rets_raw[trial - start_MDP].append(
                np.clip(r_a_hat_q_scaled_return, -1, 1)
            )
            r_r_true_scaled_rets_raw[trial - start_MDP].append(
                np.clip(r_r_true_q_scaled_return, -1, 1)
            )

    # Plot results (used for generating Figure 6)
    fig, ax = plt.subplots(1)
    x = np.array(list(range(plot_interval))) * checkpoint_every

    r_a_hat_y = r_a_hat_scaled_rets / (end_MDP - start_MDP)
    r_r_true_y = r_r_true_scaled_rets / (end_MDP - start_MDP)

    print("Mean returns when Q-learning with r_hat=A*_hat")
    print(r_a_hat_y)
    print("Mean returns when Q-learning with r_hat=r")
    print(r_r_true_y)
    print("\n")
    ax.plot(x, r_a_hat_y, label="r-hat=A*-hat", color="#bd1b1b")
    ax.plot(x, r_r_true_y, label="r-hat=r", color="#A862EA")

    ax.hlines(
        y=1,
        xmin=x[0],
        xmax=x[-1],
        colors="grey",
        linestyles="--",
        label="r-hat=A*",
        linewidth=7,
    )

    ax.plot(
        x,
        oaf_scaled_rets / (end_MDP - start_MDP),
        linestyle="--",
        color="#df8520",
        label="argmax",
    )

    ax.set_ylim(-0.6, 1.1)
    ax.set_yticks([-0.5, 0, 0.5, 1])
    ax.set_xlim(x[0], x[-1])
    ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600])
    ax.set_xlabel("Number of episodes", labelpad=15)
    ax.set_ylabel("Mean clipped scaled return", labelpad=15)

    fig.set_size_inches(12, 5)
    fig = plt.gcf()
    fig_file_name = f"{(start_MDP, end_MDP)}_MDPs_{mode}_{num_prefs}_prefs.png"
    fig.savefig(fig_file_name, dpi=300)

    # Calculate the area above the two training curves (Q learning with the true reward function, Q learning with the reward function set as A*_hat) , and then run a Wilcoxon test
    total_area = 80
    r_a_hat_y_areas = []
    r_r_true_y_areas = []

    for r_a_hat_ret, r_r_true_ret in zip(
        r_a_hat_scaled_rets_raw, r_r_true_scaled_rets_raw
    ):
        r_a_hat_y_area = total_area - np.trapz(r_a_hat_ret, dx=1)
        r_r_true_y_area = total_area - np.trapz(r_r_true_ret, dx=1)

        r_a_hat_y_areas.append(r_a_hat_y_area)
        r_r_true_y_areas.append(r_r_true_y_area)

    res1 = wilcoxon(np.asarray(r_a_hat_y_areas) - np.asarray(r_r_true_y_areas))

    print("Wilcoxon results, area_above(r_hat=A*_hat(s,a)) -  area_above(r_hat=r)")
    print("w=", res1.statistic)
    print("p=", res1.pvalue)
    print("\n")


if __name__ == "__main__":
    main()
