"""
This file enables computing the scaled return(s) under the true reward function(s) for the function g (where g is a reward function or optimal advantage function).
A scaled return of 1 means that g induced an optimal policy, 0 means that g induced a policy that performs on par with a uniformly random policy, and < 0 means
that g induced a worse than uniformly random policy.
"""

import pickle
import os
import gzip

import torch
import numpy as np

from learn_advantage.utils.argparse_utils import parse_args, ArgsType
from learn_advantage.algorithms.rl_algos import (
    value_iteration,
    build_random_policy,
    build_pi,
    get_gt_avg_return,
    iterative_policy_evaluation,
    build_pi_from_nn_feats,
    build_pi_from_feats,
)

args = parse_args(ArgsType.SCALED_RETURNS)

preference_model = args.preference_model  # how we generate prefs
preference_assum = args.preference_assum  # how we learn prefs
model = preference_model + "_" + preference_assum

all_num_prefs = [int(item) for item in args.num_prefs.split(",")]
modes = ["sigmoid" if item == "stochastic" else item for item in args.modes.split(",")]

start_MDP = args.start_MDP
end_MDP = args.end_MDP
MDP_dir = args.MDP_dir
dir_name = args.output_dir_prefix
output_dir_name = args.output_dir
extra_details = args.extra_details
extra_details_for_saving = args.extra_details_for_saving
checkpoints_of_interest = [
    int(item) for item in args.checkpoints_of_interest.split(",")
]

# there should be one function saved for each checkpoint in check_points
check_points = [1, 10, 100, 200, 500, 1000, 10000, 30000]


def get_scaled_return(
    rew_vect,
    *,
    gt_rew_vec,
    use_extended_SF,
    generalize_SF,
    learn_oaf,
    trial,
    mode,
    num_prefs,
    fp,
    env=None,
    gamma=0.999,
    force_cpu=False,
):
    """
    Given learned weights, calculates the ground truth returns collected by the learned reward functions induced policy, and scales it between
    1 and -infinity. A scaled return of 1 means an optimal policy was recovered and 0 means a random policy recovered.

    Input:
    - rew_vect: The learned weights.
    - gt_rew_vec: The ground truth reward function.
    - use_extended_SF: If false, each weight in rew_vect is the learned reward weights. Otherwise, each weight in rew_vect are (s,a) pair weights.
    - generalize_SF: If false, then we are in the tabular setting. If true then rew_vect is a neural network model.
    - learn_oaf: If true, then treat rew_vect as an optimal advantage function rather than a reward function.
    - trial,preference_model,preference_assum,mode,num_prefs: used for saving the outputted scaled return.
    - env: The environment object.
    - Gamma: The discount factor.

    Output:
    - The scaled return.
    """
    device = torch.device(
        "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    )

    if device.type != "cpu":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    if use_extended_SF:
        # derive policy from learned (s,a) weights
        if generalize_SF:
            pi = build_pi_from_nn_feats(rew_vect, env=env)
        elif learn_oaf:
            pi = build_pi_from_feats(rew_vect, env=env)
        else:
            rew_vect = rew_vect.reshape((env.height, env.width, 4))
            _, Q = value_iteration(
                rew_vec=rew_vect, gamma=gamma, env=env, extended_SF=True
            )
            pi = build_pi(Q, env=env)
    else:
        # derive policy from learned reward function
        _, Q = value_iteration(rew_vec=rew_vect, gamma=gamma, env=env)
        pi = build_pi(Q, env=env)

    if device.type != "cpu":
        end_event.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        policy_extraction_time = start_event.elapsed_time(end_event)
    else:
        policy_extraction_time = 0

    if fp is not None:
        file_parts = [
            "MDP",
            f"{trial}",
            f"{learn_oaf}",
            f"{preference_model}",
            f"{preference_assum}",
            f"mode={mode}",
            f"extended_SF={use_extended_SF}",
            f"generalize_SF={generalize_SF}",
            f"num_prefs={num_prefs}",
            "OAF.npy",
        ]
        file_name = "_".join(file_parts)
        full_path = f"{fp}{file_name}"

        if generalize_SF and use_extended_SF:
            torch.save(rew_vect.state_dict(), full_path)
        else:
            np.save(full_path, rew_vect)

    del rew_vect
    V_under_gt = iterative_policy_evaluation(pi, rew_vec=np.array(gt_rew_vec), env=env)

    avg_return = np.sum(V_under_gt) / env.n_starts

    gt_avg_return = get_gt_avg_return(gt_rew_vec=np.array(gt_rew_vec), env=env)

    # build random policy
    random_pi = build_random_policy(env=env)
    V_under_random_pi = iterative_policy_evaluation(
        random_pi, rew_vec=np.array(gt_rew_vec), env=env
    )
    random_avg_return = np.sum(V_under_random_pi) / env.n_starts

    # scale everything: f(z) = (z-x) / (y-x)
    scaled_return = (avg_return - random_avg_return) / (
        gt_avg_return - random_avg_return
    )
    return scaled_return, policy_extraction_time


def main():
    # loop through each preference data set size
    for num_prefs in all_num_prefs:
        # loop through each preference type
        for mode in modes:
            # keep track of the # of MDPs where near optimal performance was achieved by both the optimal advantage function and reward function learning methods.
            oaf_num_near_opts = np.zeros(len(check_points))
            pr_num_near_opts = np.zeros(len(check_points))

            # keep track of the scaled returns forboth the optimal advantage function and reward function learning methods.
            pr_scaled_returns = [[] for _ in range(len(check_points))]
            oaf_scaled_returns = [[] for _ in range(len(check_points))]

            # keep track of max_a(A*_hat(s,a)) for each state in each MDP
            maximum_learns_advs = [[] for _ in range(len(check_points))]

            print(
                f"Deriving and evaluating policies for MDPs {start_MDP} to {end_MDP} with advantage approximates learned from {num_prefs} preferences..."
            )

            # For each trial compute the scaled return
            for trial in range(start_MDP, end_MDP):
                # Load the MDP
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

                rew_vects_file_parts = [
                    f"{extra_details}",
                    f"{trial}_True",
                    f"{model}",
                    f"mode={mode}",
                    "extended_SF=True",
                    "generalize_SF=False",
                    f"num_prefs={num_prefs}",
                    "rew_vects.npy",
                ]

                rew_vects_file_name = "_".join(rew_vects_file_parts)
                rew_vects_fp = (
                    f"{dir_name}checkpointed_reward_vecs/{rew_vects_file_name}"
                )
                rew_vects = np.load(rew_vects_fp)

                for i, learned_rew_vect in enumerate(rew_vects):
                    if check_points[i] not in checkpoints_of_interest:
                        continue
                    # compute max_a(A*_hat(s,a))
                    rew_vect_reshaped = learned_rew_vect.reshape(
                        (env.height, env.width, 4)
                    )
                    for h in range(env.height):
                        for w in range(env.width):
                            maximum_learns_advs[i].append(
                                np.max(rew_vect_reshaped[h][w])
                            )

                    # compute scaled return for given function g, depending on if it is a reward function or an optimal advntage function
                    if model == "regret_pr":
                        scaled_return_oaf, _ = get_scaled_return(
                            learned_rew_vect,
                            gt_rew_vec=gt_rew_vec,
                            use_extended_SF=True,
                            generalize_SF=False,
                            learn_oaf=True,
                            trial=trial,
                            mode=mode,
                            num_prefs=num_prefs,
                            fp=None,
                            env=env,
                        )
                    elif model == "regret_regret":
                        scaled_return_oaf = -1

                    scaled_return_pr, _ = get_scaled_return(
                        learned_rew_vect,
                        gt_rew_vec=gt_rew_vec,
                        use_extended_SF=True,
                        generalize_SF=False,
                        learn_oaf=False,
                        trial=trial,
                        mode=mode,
                        num_prefs=num_prefs,
                        fp=None,
                        env=env,
                    )

                    if scaled_return_oaf >= 0.9:
                        oaf_num_near_opts[i] += 1
                    if scaled_return_pr >= 0.9:
                        pr_num_near_opts[i] += 1

                    oaf_scaled_returns[i].append(scaled_return_oaf)
                    pr_scaled_returns[i].append(scaled_return_pr)

            oaf_num_near_opts /= (end_MDP - start_MDP) * 100
            pr_num_near_opts /= (end_MDP - start_MDP) * 100

            if mode == "sigmoid":
                title = "Training data: " + str(num_prefs) + " stochastic preferences"
            else:
                title = "Training data: " + str(num_prefs) + " noiseless preferences"

            if "NO_ABSORBING_TRANSITIONS" in dir_name:
                title += ", no segments containing transitions from the absorbing state"
            else:
                title += (
                    ", with segments containing transitions from the absorbing state"
                )

            print("=====================")
            print(title)
            print("=====================")
            for i, oaf_num_near_opt in enumerate(oaf_num_near_opts):
                if check_points[i] not in checkpoints_of_interest:
                    continue
                print("Training for " + str(check_points[i]) + " epochs:")
                print("    Greedy A*_hat model  % near optimal: ", oaf_num_near_opt)
                print("    r_hat = A*_hat model % near optimal: ", pr_num_near_opts[i])
                print(
                    "    mean max_a(A*_hat(s,a)) across all states in all MDPs:",
                    np.mean(maximum_learns_advs[i]),
                )

                # save all scaled returns to a numpy file
                if not os.path.exists(output_dir_name):
                    os.makedirs(output_dir_name)

                base_name_parts = [
                    f"{extra_details_for_saving}",
                    f"{extra_details}",
                    f"n_epochs={check_points[i]}",
                    f"{model}",
                    f"mode={mode}",
                    f"num_prefs={num_prefs}",
                    ".npy",
                ]

                if model == "regret_pr":
                    oaf_file_name = "_".join(
                        base_name_parts[:-2]
                        + ["argmax(A*_hat(s,a))_scaled_returns", base_name_parts[-2]]
                    )
                    pr_file_name = "_".join(
                        base_name_parts[:-2]
                        + ["r_hat=A*_hat(s,a)_scaled_returns", base_name_parts[-2]]
                    )

                    np.save(output_dir_name + oaf_file_name, oaf_scaled_returns[i])
                    np.save(output_dir_name + pr_file_name, pr_scaled_returns[i])
                elif model == "regret_regret":
                    regret_file_name = "_".join(
                        base_name_parts[:-2]
                        + ["regret_scaled_returns", base_name_parts[-2]]
                    )
                    np.save(output_dir_name + regret_file_name, pr_scaled_returns[i])

            print("\n")


if __name__ == "__main__":
    main()
