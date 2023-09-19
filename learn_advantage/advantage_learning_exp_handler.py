"""
This file enables running expirements for learning g functions (where g is a reward function or optimal advantage function) for random MDPs with various preference dataset sizes.
This file is responsible for loading the preference dataset, formatting all data, and then calling the advantage_learning.train function in order to learn g with the correct parameters.
"""

import pickle
import os
import gzip
import numpy as np
import torch

from learn_advantage.utils.pref_dataset_utils import (
    augment_data,
    generate_synthetic_prefs,
    remove_absorbing_transitions,
)
from learn_advantage.utils.segment_feats_utils import get_extended_features
from learn_advantage.utils.argparse_utils import parse_args
from learn_advantage.algorithms.advantage_learning import train

args = parse_args()

force_cpu = args.force_cpu
keep_ties = args.keep_ties
n_prob_samples = args.n_prob_samples
n_prob_iters = args.n_prob_iters


def main():
    gamma = args.gamma

    mode = "sigmoid" if args.mode == "stochastic" else args.mode

    preference_model = args.preference_model  # how we generate prefs
    preference_assum = args.preference_assum  # how we learn prefs

    use_extended_SF = args.use_extended_SF
    learn_oaf = args.learn_oaf
    extra_details = args.extra_details
    seg_length = args.seg_length

    start_MDP = args.start_MDP
    end_MDP = args.end_MDP
    MDP_dir = args.MDP_dir
    all_num_prefs = [int(item) for item in args.num_prefs.split(",")]
    dont_include_absorbing_transitions = args.dont_include_absorbing_transitions
    dir_name = args.output_dir_prefix

    # These parameters provide some extra conditions primarily for debuging. They are hardcoded for now.
    use_val_set = False
    generalize_SF = args.generalize_SF = False

    args.succ_feats = None
    args.succ_q_feats = None
    args.pis = None

    if args.preference_assum == "regret":
        args.succ_feats = np.load("succ_feats_no_gt.npy", allow_pickle=True)
        args.pis = np.load("pis_no_gt.npy", allow_pickle=True)
        args.succ_q_feats = None
    else:
        args.succ_feats = args.succ_feats
        args.pis = None
        args.succ_q_feats = args.succ_q_feats

    if args.force_cpu:
        args.device = "cpu"
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_runs = 0
    # learn a reward function for each segment dataset size
    for num_prefs in all_num_prefs:
        timing_info_fp = (
            dir_name
            + "_timing_info/"
            + str(learn_oaf)
            + "_"
            + preference_model
            + "_"
            + preference_assum
            + "_mode="
            + mode
            + "_extended_SF="
            + str(use_extended_SF)
            + "_generalize_SF="
            + str(generalize_SF)
            + "_num_prefs="
            + str(num_prefs)
            + "training_times.npy"
        )
        if os.path.isfile(timing_info_fp):
            training_times = np.load(timing_info_fp)
            training_times = training_times.tolist()
        else:
            training_times = []
        print(
            "Computing approximate optimal advantages for MDPs "
            + str(start_MDP)
            + " to "
            + str(end_MDP)
            + " with "
            + str(num_prefs)
            + " preferences..."
        )
        for trial in range(start_MDP, end_MDP):
            np.random.seed(n_runs)
            n_runs += 1

            all_states = None
            all_actions = None
            # load the MDP and the dataset of segments, start/end states, and successor features
            if seg_length == 3:
                all_stored_segs = np.load(
                    gzip.GzipFile(
                        MDP_dir + "MDP_" + str(trial) + "all_trajs.npy.gz", "r"
                    )
                )
            else:
                all_stored_segs = np.load(
                    gzip.GzipFile(
                        MDP_dir
                        + "MDP_"
                        + str(trial)
                        + "all_trajs_length="
                        + str(seg_length)
                        + ".npy.gz",
                        "r",
                    )
                )

            gt_rew_vec = np.load(
                gzip.GzipFile(MDP_dir + "MDP_" + str(trial) + "gt_rew_vec.npy.gz", "r")
            )

            if preference_assum == "pr":
                succ_feats = None
                succ_q_feats = None
            elif use_extended_SF:
                succ_feats = np.load(
                    gzip.GzipFile(
                        MDP_dir
                        + "MDP_"
                        + str(trial)
                        + "sa_succ_feats_gamma="
                        + f"{gamma:.3f}"
                        + ".npy.gz",
                        "r",
                    )
                )
                succ_q_feats = None
            else:
                succ_feats = np.load(
                    gzip.GzipFile(
                        MDP_dir + "MDP_" + str(trial) + "succ_feats.npy.gz", "r"
                    )
                )
                succ_q_feats = None

            args.succ_feats = succ_feats
            args.succ_q_feats = succ_q_feats

            with open(MDP_dir + "MDP_" + str(trial) + "env.pickle", "rb") as rf:
                env = pickle.load(rf)
                if trial < 300:
                    env.generate_transition_probs()
                    env.set_custom_reward_function(gt_rew_vec)

            all_stored_segs = np.array(all_stored_segs)
            if len(all_stored_segs) > num_prefs and trial < 300:
                idx = np.random.choice(
                    np.arange(len(all_stored_segs)), num_prefs, replace=False
                )
                all_segs = all_stored_segs[idx]
            else:
                all_segs = all_stored_segs
                idx = np.random.choice(
                    np.arange(len(all_stored_segs)),
                    num_prefs - len(all_stored_segs),
                    replace=True,
                )
                all_segs = np.concatenate((all_segs, all_stored_segs[idx]))

            # get the features for each segment pair, as well as the ground truth reward and the start/end states.
            all_X, all_r, all_ses, _ = get_extended_features(
                args, all_segs, env, gt_rew_vec, seg_length=seg_length
            )

            if generalize_SF or use_val_set:
                idx = np.random.choice(
                    np.arange(len(all_stored_segs)), 2000, replace=False
                )
                all_validation_segs = all_stored_segs[idx]
                val_all_X, val_all_r, val_all_ses, _ = get_extended_features(
                    args, all_validation_segs, env, gt_rew_vec, seg_length=seg_length
                )

                val_X, val_y, _ = generate_synthetic_prefs(
                    args,
                    pr_X=val_all_X,
                    rewards=val_all_r,
                    sess=val_all_ses,
                    actions=all_actions,
                    states=all_states,
                    mode=mode,
                    gt_rew_vec=np.array(gt_rew_vec),
                    env=env,
                )
                val_X, val_y = augment_data(val_X, val_y, "arr")
            else:
                val_X = None
                val_y = None

            if use_extended_SF and dont_include_absorbing_transitions:
                all_X, all_r, all_ses = remove_absorbing_transitions(
                    args,
                    all_X=all_X,
                    all_r=all_r,
                    all_ses=all_ses,
                    env=env,
                    gt_rew_vec=gt_rew_vec,
                    seg_length=seg_length,
                    all_stored_segs=all_stored_segs,
                )

            # generate synthetic preferences using the ground truth reward/value function.
            pr_X, synth_max_y, _ = generate_synthetic_prefs(
                args,
                pr_X=all_X,
                rewards=all_r,
                sess=all_ses,
                actions=all_actions,
                states=all_states,
                mode=mode,
                gt_rew_vec=np.array(gt_rew_vec),
                env=env,
            )

            # augment the dataset by swapping preferences and segment pairs
            aX, ay = augment_data(pr_X, synth_max_y, "arr")

            # learn a reward function from the dataset
            plot_loss = False
            rew_vects, _, _, training_time = train(
                aX=aX,
                ay=ay,
                args=args,
                plot_loss=plot_loss,
                env=env,
                validation_X=val_X,
                validation_y=val_y,
                check_point=True,
            )
            training_times.append(training_time)

            if not os.path.exists(dir_name + "checkpointed_reward_vecs"):
                os.makedirs(dir_name + "checkpointed_reward_vecs")

            rew_vects_file_name = "_".join(
                [
                    f"{extra_details}",
                    f"{trial}",
                    f"{learn_oaf}",
                    f"{preference_model}",
                    f"{preference_assum}",
                    f"mode={mode}",
                    f"extended_SF={use_extended_SF}",
                    f"generalize_SF={generalize_SF}",
                    f"num_prefs={num_prefs}",
                    "rew_vects.npy",
                ]
            )
            rew_vects_path = "/".join(
                [dir_name + "checkpointed_reward_vecs", rew_vects_file_name]
            )
            np.save(rew_vects_path, rew_vects)

            if not os.path.exists(dir_name + "_timing_info"):
                os.makedirs(dir_name + "_timing_info")

            training_times_file_name = "_".join(
                [
                    f"{extra_details}",
                    f"{learn_oaf}",
                    f"{preference_model}",
                    f"{preference_assum}",
                    f"mode={mode}",
                    f"extended_SF={use_extended_SF}",
                    f"generalize_SF={generalize_SF}",
                    f"num_prefs={num_prefs}",
                    "training_times.npy",
                ]
            )

            training_times_path = "/".join(
                [dir_name + "_timing_info", training_times_file_name]
            )

            np.save(training_times_path, training_times)


if __name__ == "__main__":
    main()
