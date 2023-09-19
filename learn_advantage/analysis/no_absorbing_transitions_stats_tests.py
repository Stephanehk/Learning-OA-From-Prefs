"""
This file enables runs the Wilcoxon paired signed rank tests on the data generated when learning A*_hat(s,a)
with and without transitions from the absorbing state. These tests provide statistical analysis for data in Figures 3,4,10, and 11.
"""

import pickle

import numpy as np
from scipy.stats import wilcoxon

from learn_advantage.utils.argparse_utils import parse_args, ArgsType

args = parse_args(ArgsType.STATS_TEST)


def main():
    preference_model = args.preference_model  # how we generate prefs
    preference_assum = args.preference_assum  # how we learn prefs
    model = preference_model + "_" + preference_assum

    all_num_prefs = [int(item) for item in args.num_prefs.split(",")]
    modes = [
        "sigmoid" if item == "stochastic" else item for item in args.modes.split(",")
    ]

    start_MDP = args.start_MDP
    end_MDP = args.end_MDP
    MDP_dir = args.MDP_dir

    output_dir_name = args.output_dir
    extra_details = args.extra_details
    extra_details_for_saving = args.extra_details_for_saving
    number_of_epochs = args.number_of_epochs
    check_points = [1, 10, 100, 200, 500, 1000, 10000, 30000]

    # loop through each preference data set size
    for num_prefs in all_num_prefs:
        # loop through each preference type
        for mode in modes:
            # Save paired data, where first item in pair is from learning with transitions from the absorbing state and second item is without such transitions
            paired_maximum_learns_advs = []
            paired_oaf_scaled_rets = []
            paired_pr_scaled_rets = []
            # loop through the directory containing expirements with and without transitions from the absorbing state
            for dir_name in (
                "data/results/WITH_ABSORBING_TRANSITIONS_",
                "data/results/NO_ABSORBING_TRANSITIONS_",
            ):
                base_name_parts = [
                    f"{extra_details_for_saving}",
                    f"{extra_details}",
                    f"n_epochs={number_of_epochs}",
                    f"{model}",
                    f"mode={mode}",
                    f"num_prefs={num_prefs}",
                    ".npy",
                ]

                if model == "regret_pr":
                    oaf_file_name = (
                        "_".join(
                            base_name_parts[:-2]
                            + [
                                "argmax(A*_hat(s,a))_scaled_returns",
                                base_name_parts[-2],
                            ]
                        )
                        + ".npy"
                    )

                    oaf_scaled_returns = np.load(output_dir_name + oaf_file_name)

                else:
                    print(
                        "THIS TEST IS CURRENTLY NOT SUPPORTED FOR THE REGRET BASED REWARD LEARNING MODEL!"
                    )
                paired_oaf_scaled_rets.append(oaf_scaled_returns)
                paired_pr_scaled_rets.append(paired_pr_scaled_rets)

                maximum_learns_advs = []
                # For each trial compute max_a(A*_hat(s,a))
                for trial in range(start_MDP, end_MDP):
                    with open(MDP_dir + "MDP_" + str(trial) + "env.pickle", "rb") as rf:
                        env = pickle.load(rf)

                    file_parts = [
                        f"{extra_details}",
                        f"{trial}_True",
                        f"{model}",
                        f"mode={mode}",
                        "extended_SF=True",
                        "generalize_SF=False",
                        f"num_prefs={num_prefs}",
                        "rew_vects.npy",
                    ]

                    file_name = "_".join(file_parts)
                    fp = f"{dir_name}checkpointed_reward_vecs/{file_name}"

                    rew_vects = np.load(fp)
                    rew_vect = rew_vects[check_points.index(number_of_epochs)]

                    # compute max_a(A*_hat(s,a))
                    rew_vect_reshaped = rew_vect.reshape((env.height, env.width, 4))
                    for h in range(env.height):
                        for w in range(env.width):
                            maximum_learns_advs.append(np.max(rew_vect_reshaped[h][w]))

                paired_maximum_learns_advs.append(maximum_learns_advs)

            # Wilcoxon tests from Figure 4/10
            maximum_learns_advs_wilcoxon_res = wilcoxon(
                np.array(paired_maximum_learns_advs[0])
                - np.array(paired_maximum_learns_advs[1])
            )
            print(
                "Wilcoxon paired signedrank tests between max_a(A*_hat(s,a)) when learning with versus without transitions from the absorbing state"
            )
            print("    w=", maximum_learns_advs_wilcoxon_res.statistic)
            print("    p=", maximum_learns_advs_wilcoxon_res.pvalue)
            print("\n")

            # Wilcoxon tests from Figure 3/11
            oaf_scaled_ret_diff = np.array(paired_oaf_scaled_rets[0]) - np.array(
                paired_oaf_scaled_rets[1]
            )

            if sum(oaf_scaled_ret_diff) == 0:
                print(
                    "Cannot perform Wilcoxon test for Greedy A*_hat model, all sample differences are 0. Try increasing the sample size."
                )
            else:
                oaf_scaled_rets_wilcoxon_res = wilcoxon(oaf_scaled_ret_diff)
                print(
                    "Wilcoxon paired signedrank tests between scaled returns of Greedy A*_hat model when learning with versus without transitions from the absorbing state"
                )
                print("    w=", oaf_scaled_rets_wilcoxon_res.statistic)
                print("    p=", oaf_scaled_rets_wilcoxon_res.pvalue)
                print("\n")

            pr_scaled_ret_diff = np.array(paired_pr_scaled_rets[0]) - np.array(
                paired_pr_scaled_rets[1]
            )

            if sum(pr_scaled_ret_diff) == 0:
                print(
                    "Cannot perform Wilcoxon test for r_hat = A*_hat model, all sample differences are 0. Try increasing the sample size."
                )
            else:
                pr_scaled_rets_wilcoxon_res = wilcoxon(pr_scaled_ret_diff)
                print(
                    "Wilcoxon paired signedrank tests between scaled returns of r_hat = A*_hat model when learning with versus without transitions from the absorbing state"
                )
                print("    w=", pr_scaled_rets_wilcoxon_res.statistic)
                print("    p=", pr_scaled_rets_wilcoxon_res.pvalue)
                print("\n")


if __name__ == "__main__":
    main()
