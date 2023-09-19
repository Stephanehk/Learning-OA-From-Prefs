import numpy as np

# matplotlib.use('TkAgg')
from torch.utils.data import Dataset

from learn_advantage.utils.utils import get_gt_regret, sigmoid
from learn_advantage.utils.segment_feats_utils import get_extended_features


class CustomDataset(Dataset):
    def __init__(self, dataX, dataY):
        self.dataX = dataX
        self.dataY = dataY

    def __len__(self):
        return self.dataX.size(0)

    def __getitem__(self, idx):
        return self.dataX[idx], self.dataY[idx]


def generate_synthetic_prefs(
    args,
    *,
    pr_X,
    rewards,
    sess,
    actions,
    states,
    mode,
    gt_rew_vec=None,
    env=None,
    regret_weights=None,
):
    """
    Generates synthetic preferences using the model specified by preference_assum

    Input:
        - pr_X: a list of each segment pairs features
        - rewards: a list of each segment pairs partial return
        - sess: a list of the the start and end states for each segment pair
        - actions/states: a list of state action pairs for each segment, only used for stochastic MDPs
        - mode: sigmoid (for stochastic preferences) or deterministic (for error-free preferences)
        - gt_rew_vec: the ground truth reward vector, None if using the default delivery domain
        - env: the environment, None if using the default delivery domain

    Output:
        - a list of syntheticallt generated preferences and their corresponding preference pair features
    """
    synth_y = []
    non_redundent_pr_X = []
    expected_returns = []
    if gt_rew_vec is None:
        gt_rew_vec = [-1, 50, -50, 1, -1, -2]
    index = 0

    # quick error handling
    if actions is None:
        actions = [None for i in range(len(pr_X))]
        states = [None for i in range(len(pr_X))]

    assert len(sess) == len(pr_X) == len(rewards)
    precomputed_vf = False

    for r, x, ses in zip(rewards, pr_X, sess):
        x = list(x)
        if args.preference_model == "pr" and args.preference_assum == "pr":
            x_f = [list(x[0][0:6]), list(x[1][0:6])]
            x_orig = [list(x[0]), list(x[1])]
        # change x to include start end state for each segectory
        if args.preference_model == "regret" and args.preference_assum == "regret":
            if args.include_actions:
                x[0] = list(x[0])
                x[0].extend(actions[index][0])
                x[0].extend(
                    np.array(states[index][0][0 : len(states[index][0]) - 1]).flatten()
                )

                x[1] = list(x[1])
                x[1].extend(actions[index][1])
                x[1].extend(
                    np.array(states[index][1][0 : len(states[index][1]) - 1]).flatten()
                )

                x_f = [x[0], x[1]]
                x_orig = [list(x[0]), list(x[1])]

            else:
                x[0] = list(x[0])
                x[0].extend([ses[0][0][0], ses[0][0][1], ses[0][1][0], ses[0][1][1]])

                x[1] = list(x[1])
                x[1].extend([ses[1][0][0], ses[1][0][1], ses[1][1][0], ses[1][1][1]])
                x_f = [x[0], x[1]]
                x_orig = [list(x[0]), list(x[1])]
        if args.preference_model == "pr" and args.preference_assum == "regret":
            if args.include_actions:
                raise ValueError("Unsupported use of actions.")
            else:
                x[0] = list(x[0])
                x[0].extend([ses[0][0][0], ses[0][0][1], ses[0][1][0], ses[0][1][1]])

                x[1] = list(x[1])
                x[1].extend([ses[1][0][0], ses[1][0][1], ses[1][1][0], ses[1][1][1]])
                x_f = [x[0], x[1]]
                x_orig = [list(x[0]), list(x[1])]
        if args.preference_model == "regret" and args.preference_assum == "pr":
            x_f = [list(x[0]), list(x[1])]
            x_orig = [list(x[0]), list(x[1])]

        t1_ss = [int(ses[0][0][0]), int(ses[0][0][1])]
        t1_es = [int(ses[0][1][0]), int(ses[0][1][1])]

        t2_ss = [int(ses[1][0][0]), int(ses[1][0][1])]
        t2_es = [int(ses[1][1][0]), int(ses[1][1][1])]

        if mode == "sigmoid":
            if args.preference_model == "regret":
                r1_er, r2_er = get_gt_regret(
                    args,
                    precomputed_vf=precomputed_vf,
                    x=x,
                    r=r,
                    t1_ss=t1_ss,
                    t1_es=t1_es,
                    t2_ss=t2_ss,
                    t2_es=t2_es,
                    actions=actions[index],
                    states=states[index],
                    gt_rew_vec=gt_rew_vec,
                    env=env,
                    regret_weights=regret_weights,
                )
                precomputed_vf = True

            if args.preference_model == "pr" and not args.keep_ties and r[1] == r[0]:
                continue

            if (
                args.preference_model == "regret"
                and not args.keep_ties
                and r1_er == r2_er
            ):
                continue

            for _ in range(args.n_prob_samples):
                if args.preference_model == "pr":
                    r1_prob = sigmoid((r[0] - r[1]) / 1)
                    r2_prob = sigmoid((r[1] - r[0]) / 1)
                elif args.preference_model == "regret":
                    r1_prob = sigmoid((r1_er - r2_er) / 1)
                    r2_prob = sigmoid((r2_er - r1_er) / 1)
                num = np.random.choice([1, 0], p=[r1_prob, r2_prob])
                if num == 1:
                    pref = [1, 0]
                elif num == 0:
                    pref = [0, 1]
                synth_y.append(pref)
                non_redundent_pr_X.append(x_orig)
        else:
            if args.preference_model == "regret":
                r1_er, r2_er = get_gt_regret(
                    args,
                    precomputed_vf=precomputed_vf,
                    x=x,
                    r=r,
                    t1_ss=t1_ss,
                    t1_es=t1_es,
                    t2_ss=t2_ss,
                    t2_es=t2_es,
                    actions=actions[index],
                    states=states[index],
                    gt_rew_vec=gt_rew_vec,
                    env=env,
                    regret_weights=regret_weights,
                )
                precomputed_vf = True

                pref = get_pref([r1_er, r2_er], include_eps=False)
            else:
                pref = get_pref(r, include_eps=False)

            if pref == [0.5, 0.5] and not args.keep_ties:
                continue

            if args.preference_model == "regret":
                expected_returns.append([r1_er, r2_er])
            synth_y.append(pref)
            non_redundent_pr_X.append(x_orig)
        index += 1
    return non_redundent_pr_X, synth_y, expected_returns


def remove_absorbing_transitions(
    args, *, all_X, all_r, all_ses, env, gt_rew_vec, seg_length, all_stored_segs
):
    """
    Given a list of segment pairs and their corresponding statistics, removes all segment pairs that contain transitions from the absorbing state

    Input:
    - all_X: a list of segment pair features
    - all_r: a list of the partial returns for each segment in each segment pair
    - all_ses: a list of the start and end states for each segment in each segment pair
    - gt_rew_vec: the true reward vector that we want to recover using preferences
    - seg_length: the number of transitions in each segment

    Output:
    - all_X: a list of segment pair features with all segments containing transitions from the absorbing state having been replaced
    - all_r: a list of the partial returns for each segment in each segment pair with all segments containing transitions from the absorbing state having been replaced
    - all_ses: a list of the start and end states for each segment in each segment pair with all segments containing transitions from the absorbing state having been replaced

    """
    inices2delete = []
    for x_pair_i, x_pair in enumerate(all_X):
        if sum(x_pair[0]) < seg_length or sum(x_pair[1]) < seg_length:
            inices2delete.append(x_pair_i)

    all_X = [v for i, v in enumerate(all_X) if i not in frozenset(inices2delete)]
    all_r = [v for i, v in enumerate(all_r) if i not in frozenset(inices2delete)]
    all_ses = [v for i, v in enumerate(all_ses) if i not in frozenset(inices2delete)]

    if len(inices2delete) * 5 < len(all_stored_segs):
        idx = np.random.choice(
            np.arange(len(all_stored_segs)), len(inices2delete) * 5, replace=False
        )
    else:
        idx = np.random.choice(
            np.arange(len(all_stored_segs)), len(all_stored_segs), replace=True
        )

    all_additional_segs = all_stored_segs[idx]
    all_add_X, all_add_r, all_add_ses, _ = get_extended_features(
        args, all_additional_segs, env, gt_rew_vec, seg_length=seg_length
    )
    added_pairs = 0
    for x_pair_i, x_pair in enumerate(all_add_X):
        if not (
            sum(x_pair[0]) < seg_length or sum(x_pair[1]) < seg_length
        ) and added_pairs < len(inices2delete):
            added_pairs += 1
            all_X.append(x_pair)
            all_r.append(all_add_r[x_pair_i])
            all_ses.append(all_add_ses[x_pair_i])

    for x_pair_i, x_pair in enumerate(all_X):
        if sum(x_pair[0]) < seg_length or sum(x_pair[1]) < seg_length:
            raise ValueError("ERROR IN DELETING MULTI-LENGTH SEGMENTS")
    return all_X, all_r, all_ses


def get_pref(arr, include_eps=True):
    """
    Given a segment pair stastic (partial return, regret, etc.), returns the error-free preference for that segment pair
    """
    if include_eps:
        arr[0] = np.round(arr[0], 1)
        arr[1] = np.round(arr[1], 1)

    if arr[0] > arr[1]:
        res = [1, 0]
    elif arr[1] > arr[0]:
        res = [0, 1]
    else:
        res = [0.5, 0.5]
    return res


def augment_data(X, Y, ytype="scalar"):
    """
    Augments the preference dataset by flipping the segment pairs and the corresponding preferences

    Input:
    - X: a list of segment pair reward features
    - Y: a list of preferences for each segment pair
    - ytype: if ytype == "scalar", then preferences are scalar values (0 means the first segment is preffered, 1 means the second preference is preffered, 0.5 means they are equally preferred)
             otherwise, the preferences are represented as arrays ([1,0] means the first segment is preffered, [0,1] means the second preference is preffered, [0.5,0.5] means they are equally preferred)

    Output:
    - aX: the augmented list of segment pair reward features
    - aY: the augmented list of preferences
    """

    aX = []
    ay = []
    for x, y in zip(X, Y):
        aX.append(x)
        ay.append(y)

        neg_x = [x[1], x[0]]
        aX.append(neg_x)
        if ytype == "scalar":
            ay.append(1 - y)
        else:
            ay.append([y[1], y[0]])
    return aX, ay
