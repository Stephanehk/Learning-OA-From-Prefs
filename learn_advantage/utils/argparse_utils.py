import argparse
import enum


class ArgsType(enum.IntEnum):
    DEFAULT = enum.auto()
    FIG_5 = enum.auto()
    FIG_6 = enum.auto()
    STATS_TEST = enum.auto()
    SCALED_RETURNS = enum.auto()


def parse_args(arg: ArgsType = ArgsType.DEFAULT) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch RL trainer")

    parser.add_argument(
        "--start_MDP",
        default=100,
        type=int,
        help="First MDP index indicating which MDP to learn a reward function for. Reward functions will be learned for MDPs from index start_MDP (inclusive) to end_MDP (exclusive).",
    )
    parser.add_argument(
        "--end_MDP",
        default=200,
        type=int,
        help="Last MDP index indicating which MDP to learn a reward function for. Reward functions will be learned for MDPs from index start_MDP (inclusive) to end_MDP (exclusive).",
    )
    parser.add_argument(
        "--preference_model",
        type=str,
        default="regret",
        help="preference model for how we generate synthetic preferences (pr for partial return model, er for regret model)",
    )
    parser.add_argument(
        "--preference_assum",
        type=str,
        default="regret",
        help="preference model for how we learn a reward function from preferences (pr for partial return model, er for regret model)",
    )

    parser.add_argument(
        "--num_prefs",
        default="300,1000,3000,10000,30000,100000",
        type=str,
        help="A delimited list containing preference dataset sizes, each of which will be used when learnin a reward function for each MDP.",
    )
    parser.add_argument(
        "--output_dir_prefix",
        default="data/results/",
        type=str,
        help="Prefix used to save outputted data. Learned reward vectors will be stored in {output_dir_prefix}_checkpointed_reward_vecs/ and timing info in {output_dir_prefix}_timing_info/",
    )

    parser.add_argument(
        "--extra_details",
        type=str,
        default="",
        help="Extra information included when loading training info",
    )

    parser.add_argument(
        "--preference_condition",
        type=str,
        default="",
        help="Condition which human subjects were shown when eliciting preferences (REGRET_UI, PARTIAL RETURN_UI, or empty quotes for NO_STATS_UI)",
    )

    if arg == ArgsType.DEFAULT:
        parser.add_argument(
            "--keep_ties",
            action="store_true",
            default=True,
            help="Keep indifferent preferences in preference dataset",
        )
        parser.add_argument(
            "--n_prob_samples",
            default=1,
            type=int,
            help="Number of times a preference label is sampled from each segment pair when using a stochastic preference model",
        )
        parser.add_argument(
            "--n_prob_iters",
            default=30,
            type=int,
            help="Number of trials for learning a reward function when using a stochastic preference model",
        )
        parser.add_argument(
            "--gamma",
            default=0.999,
            type=float,
            help="Discount factor for preference labeling and reward learning/evaluation",
        )

        parser.add_argument(
            "--N_ITERS", default=5000, type=int, help="Number of training iterations"
        )

        parser.add_argument(
            "--force_cpu",
            action="store_true",
            default=False,
            help="Force torch.device to CPU (use when running on Mastadon cluster)",
        )

        parser.add_argument(
            "--use_extended_SF",
            action="store_true",
            default=True,
            help="If true, we use features for each (s,a) pair in the MDP rather than for each reward component",
        )
        parser.add_argument(
            "--learn_oaf",
            action="store_true",
            default=True,
            help="If true, we learn an optimal advantage function rather than a reward function (future work on this to come)",
        )

        parser.add_argument(
            "--seg_length",
            default=3,
            type=int,
            help="Number of transitions to include in each segment of the traning dataset",
        )
        parser.add_argument(
            "--dont_include_absorbing_transitions",
            action="store_true",
            default=False,
            help="If true, we do not include any transitions from the absorbing state which have an enforced advantage/reward of 0.",
        )

    if arg == ArgsType.FIG_5:
        parser.add_argument(
            "--MDP_dir",
            default="data/input/fig_5_random_MDPs/",
            type=str,
            help="The directory name containing all training data from randomly generated MDPs (ie: MDP files, preference datasets, etc.)",
        )
        parser.add_argument(
            "--all_extra_details",
            type=str,
            default="two_trans,single_trans",
            help="Extra information included when loading training info",
        )
    else:
        parser.add_argument(
            "--MDP_dir",
            default="data/input/random_MDPs/",
            type=str,
            help="The directory name containing all training data from randomly generated MDPs (ie: MDP files, preference datasets, etc.)",
        )

    if arg == ArgsType.FIG_6:
        parser.add_argument(
            "--n_episodes",
            type=int,
            default=1600,
            help="Number of episodes to run Q learning for",
        )
        parser.add_argument(
            "--checkpoint_freq",
            type=int,
            default=20,
            help="The frequency at which the curent policy is evaluated during Q learning (given in episodes)",
        )

    if arg == ArgsType.SCALED_RETURNS:
        parser.add_argument(
            "--checkpoints_of_interest",
            type=str,
            default="1000",
            help="A delimited list containing the checkpoints we wish to load the learned function from.",
        )

    if arg == ArgsType.STATS_TEST:
        parser.add_argument(
            "--number_of_epochs",
            type=int,
            default=1000,
            help="Loads the weight vector that was saved after training for this many epochs. Should be one of the followin: [1,10,100,200,500,1000,10000,30000]",
        )

    if arg in {ArgsType.SCALED_RETURNS, ArgsType.STATS_TEST}:
        parser.add_argument(
            "--output_dir",
            default="data/results/scaled_returns/",
            type=str,
            help="Path to save outputted data.",
        )

        parser.add_argument(
            "--extra_details_for_saving",
            type=str,
            default="",
            help="Extra information included when saving training info",
        )

    if arg in {ArgsType.FIG_5, ArgsType.SCALED_RETURNS, ArgsType.STATS_TEST}:
        parser.add_argument(
            "--modes",
            type=str,
            default="deterministic,stochastic",
            help="A delimited list containing the preference types (ie: either deterministic, stochastic, or both)",
        )

    if arg in {ArgsType.DEFAULT, ArgsType.FIG_6, ArgsType.STATS_TEST}:
        parser.add_argument("--LR", default=0.5, type=float, help="Learning rate")

        parser.add_argument(
            "--mode",
            default="deterministic",
            type=str,
            help="Either deterministic (for error-free synthetic preference dataset), sigmoid (for stochastic synthetic preference dataset), or deterministic_user_data (for the human preference dataset)",
        )

    args = parser.parse_args()

    # these are arguments hardcode for now but can be changed for testing
    args.include_actions = False
    args.oa_shift = 0
    args.generalize_SF = False

    return args
