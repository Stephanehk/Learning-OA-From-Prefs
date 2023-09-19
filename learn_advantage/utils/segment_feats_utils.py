import numpy as np
import torch

from learn_advantage.utils.utils import is_in_blocked_area
from learn_advantage.env.grid_world import GridWorldEnv


def get_extended_features(args, all_segs, env=None, gt_rew_vec=None, seg_length=3):
    """
    Given a dataset of segment pairs, this function finds the features for each segment pair. If args.use_extended_SF = False, then these are the
    reward features. Otherwise, these will be the (s,a) pair features.

    Input:
     - all_segs: A list of segment pairs formatted as [[[s1,a1,a2,..], [s1,a1,a2,...]], [[s1,a1,a2,..], [s1,a1,a2,...]], ...]
     - env: A GridWorld object containing information about the given MDP. If None, the delivery domain instantation MDP is used.
     - gt_rew_vec: The ground truth reward function. If None, the delivery domain instantation MDP's reward function is used.
     - seg_length: The segment length.

     Output:
     - all_X: The features for each segment in each segment pair.
     - all_r: The ground truth partial return for each segment pair.
     - all_ses: The start and end states for each segment in each segment pair.
     - visited_states: A list of states that are covered by the inputted segment dataset.
    """
    if env is None:
        env = GridWorldEnv("2021-07-29_sparseboard2-notrap")
        env.board = np.array(env.board)
        if gt_rew_vec is None:
            gt_rew_vec = np.array([-1, 50, -50, 1, -1, -2])

    all_X = []
    all_r = []
    all_ses = []
    visited_states = []

    # TODO: CANNOT USE THIS find_reward_features METHOD WITH STOCHASTIC TRANSITIONS
    for seg_pair in all_segs:
        _, phi1 = find_reward_features(
            seg_pair[0],
            env,
            use_extended_SF=args.use_extended_SF,
            gamma=args.gamma,
            seg_length=seg_length,
        )
        _, phi2 = find_reward_features(
            seg_pair[1],
            env,
            use_extended_SF=args.use_extended_SF,
            gamma=args.gamma,
            seg_length=seg_length,
        )

        while len(seg_pair[0]) < seg_length + 1:
            seg_pair[0].extend([[0, 0]])

        while len(seg_pair[1]) < seg_length + 1:
            seg_pair[1].extend([[0, 0]])

        sa_list_1, end_state1 = get_sa_list(seg_pair[0], env, seg_length=seg_length)
        sa_list_2, end_state2 = get_sa_list(seg_pair[1], env, seg_length=seg_length)

        sa_list_1 = np.array(sa_list_1)
        sa_list_2 = np.array(sa_list_2)

        visited_states.extend(sa_list_1[:, :2])
        visited_states.extend(sa_list_2[:, :2])

        if args.use_extended_SF:
            _, phi1_rew = find_reward_features(
                seg_pair[0],
                env,
                use_extended_SF=False,
                gamma=args.gamma,
                seg_length=seg_length,
            )
            _, phi2_rew = find_reward_features(
                seg_pair[1],
                env,
                use_extended_SF=False,
                gamma=args.gamma,
                seg_length=seg_length,
            )
            all_r.append(
                [np.dot(gt_rew_vec, phi1_rew[0:6]), np.dot(gt_rew_vec, phi2_rew[0:6])]
            )
            if args.generalize_SF:
                all_X.append(
                    [
                        create_sa_embedding(sa_list_1, env.height, env.width),
                        create_sa_embedding(sa_list_2, env.height, env.width),
                    ]
                )
            else:
                all_X.append([phi1, phi2])
        else:
            all_r.append([np.dot(gt_rew_vec, phi1[0:6]), np.dot(gt_rew_vec, phi2[0:6])])
            all_X.append([phi1, phi2])

        all_ses.append([[sa_list_1[0][:2], end_state1], [sa_list_2[0][:2], end_state2]])

    return all_X, all_r, all_ses, visited_states


def get_action_feature(x, y, a, env=None):
    """
    Returns the (s,a) pair feature for a given state, (x,y)
    """
    if env is None:
        arr = np.zeros((10, 10, 4))
    else:
        arr = np.zeros((env.height, env.width, 4))
    arr[x][y][a] = 1
    return np.ravel(arr)


def get_state_feature(x, y, env=None):
    """
    Returns the state feature for a given state, (x,y)
    """
    b = env.board

    reward_feature = np.zeros(6)
    if b[x][y] == 0:
        reward_feature[0] = 1
    elif b[x][y] == 1:
        # flag
        reward_feature[1] = 1
    elif b[x][y] == 2:
        # house
        pass
    elif b[x][y] == 3:
        # sheep
        reward_feature[2] = 1
    elif b[x][y] == 4:
        # coin
        reward_feature[0] = 1
        reward_feature[3] = 1
    elif b[x][y] == 5:
        # road block
        reward_feature[0] = 1
        reward_feature[4] = 1
    elif b[x][y] == 6:
        # mud area
        reward_feature[5] = 1
    elif b[x][y] == 7:
        # mud area + flag
        reward_feature[1] = 1
    elif b[x][y] == 8:
        # mud area + house
        pass
    elif b[x][y] == 9:
        # mud area + sheep
        reward_feature[2] = 1
    elif b[x][y] == 10:
        # mud area + coin
        reward_feature[5] = 1
        reward_feature[3] = 1
    elif b[x][y] == 11:
        # mud area + roadblock
        reward_feature[5] = 1
        reward_feature[4] = 1
    return reward_feature


def create_sa_embedding(sa_list, height, width):
    """
    Given a list of states and action indices in a segment, returns a one-hot embedding for both the state and action

    Input:
    - sa_list:  a list of states and action indices in the segment

    Output:
    - sa_embedding:  a one-hot embedding of states and actions in the segment

    """
    sa_embeddings = []
    for sa in sa_list:
        grid = np.zeros((height, width))
        grid[sa[0]][sa[1]] = 1
        state_embedding = grid.flatten()
        action_embedding = np.zeros(4)
        action_embedding[sa[2]] = 1

        sa_embeddings.append(np.concatenate((state_embedding, action_embedding)))

    return sa_embeddings


def format_y(Y, ytype="scalar"):
    """
    If the inputted list of preferences are scalar values, converts them to arrays. Otherwise does nothing.

    Input:
    - Y: a list of preferences for each segment pair
    - ytype: if ytype == "scalar", then preferences are scalar values (0 means the first segment is preffered, 1 means the second preference is preffered, 0.5 means they are equally preferred)
             otherwise, the preferences are represented as arrays ([1,0] means the first segment is preffered, [0,1] means the second preference is preffered, [0.5,0.5] means they are equally preferred)
    Output:
    - formatted_y: a tensor containing preferences represented as arrays

    """
    formatted_y = []
    if ytype == "scalar":
        for y in Y:
            if y == 0:
                formatted_y.append(np.array([1, 0]))
            elif y == 1:
                formatted_y.append(np.array([0, 1]))
            elif y == 0.5:
                formatted_y.append(np.array([0.5, 0.5]))
    else:
        formatted_y = Y
    return torch.tensor(formatted_y, dtype=torch.float)


def format_X(X):
    """
    Converts X, a list of segment pair reward features, to a float tensor
    """
    return torch.tensor(X, dtype=torch.float)


def format_X_pr(X):
    """
    Returns a list of difference in partial return values for segment pairs

    Input:
    - X is a list of segment pair stastics where X[i] = [difference in partial return, difference in start state value, difference in end state value]
    Output
    - formatted_X: a list of difference in partial returns for segment pairs
    """
    formatted_X = []
    for x in X:
        formatted_X.append([x[0]])
    return torch.tensor(formatted_X, dtype=torch.float)


def format_X_regret(X):
    """
    Returns a list of difference in regret values for segment pairs

    Input:
    - X is a list of segment pair stastics where X[i] = [difference in partial return, difference in start state value, difference in end state value]
    Output
    - formatted_X: a list of difference in regret values for segment pairs
    """

    formatted_X = []
    for x in X:
        formatted_X.append([x[0] + x[1] - x[2]])
    return torch.tensor(formatted_X, dtype=torch.float)


def format_X_full(X):
    """
    Converts X to a float tensor
    """
    return torch.tensor(X, dtype=torch.float)


def get_sa_list(seg, env, seg_length=3):
    """
    Given a segment, returns a list of states and action indices in the segment. For a segment of length 3, this would be: [[s1, a1], [s2, a2], [s2, a3]]

    Input:
    - seg: the inputted segment, represented as the segments start state and the sequence of actions that follows
    - env: the environment object
    - seg_length: the segment length, measured as number of transitions

    Output:
    - sa_list:  a list of states and action indices in the segment
    """
    seg_ts_x = seg[0][0]
    seg_ts_y = seg[0][1]

    sa_list = []

    for i in range(1, seg_length + 1):
        sa_list.append([seg_ts_x, seg_ts_y, GridWorldEnv.find_action_index(seg[i])])

        if (
            seg_ts_x + seg[i][0] >= 0
            and seg_ts_x + seg[i][0] < len(env.board)
            and seg_ts_y + seg[i][1] >= 0
            and seg_ts_y + seg[i][1] < len(env.board[0])
            and not is_in_blocked_area(
                seg_ts_x + seg[i][0], seg_ts_y + seg[i][1], env.board
            )
        ):
            if not env.is_terminal(seg_ts_x, seg_ts_y):
                seg_ts_x += seg[i][0]
                seg_ts_y += seg[i][1]

    return sa_list, [seg_ts_x, seg_ts_y]


def find_reward_features(seg, env, use_extended_SF=False, gamma=1, seg_length=3):
    """
    Given a segment, returns the some of reward features for that segment

    Input:
    - seg: the inputted segment, represented as the segments start state and the sequence of actions that follows
    - env: the environment object
    - use_extended_SF: if true, then there is one reward feature for each possible state action pair and each component of the reward function. If false, then there is one reward feature per component of the reward function only.
    - gamma: the discount factor
    - seg_length: the segment length, measured as number of transitions

    Output:
    - phi_dis: the discounted sum of reward features for the inputted segment, where the discount factor is gamma
    - phi: the undiscounted sum of reward features for the inputted segment
    """
    seg_ts_x = seg[0][0]
    seg_ts_y = seg[0][1]

    prev_x = seg_ts_x
    prev_y = seg_ts_y

    if use_extended_SF:
        phi = np.zeros((4 * env.width * env.height))
        phi_dis = np.zeros((4 * env.width * env.height))
    else:
        phi = np.zeros(6)
        phi_dis = np.zeros(6)

    # TODO: ADD DATACLASS (@dataclass)
    for i in range(1, seg_length + 1):
        if env.is_terminal(seg_ts_x, seg_ts_y):
            continue
        if (
            seg_ts_x + seg[i][0] >= 0
            and seg_ts_x + seg[i][0] < len(env.board)
            and seg_ts_y + seg[i][1] >= 0
            and seg_ts_y + seg[i][1] < len(env.board[0])
            and not is_in_blocked_area(
                seg_ts_x + seg[i][0], seg_ts_y + seg[i][1], env.board
            )
        ):
            seg_ts_x += seg[i][0]
            seg_ts_y += seg[i][1]
        if (seg_ts_x, seg_ts_y) != (prev_x, prev_y):
            dis_state_sf = (gamma ** (i - 1)) * get_state_feature(
                seg_ts_x, seg_ts_y, env
            )
            state_sf = get_state_feature(seg_ts_x, seg_ts_y, env)
        # check if we are at terminal state
        elif env.is_terminal(seg_ts_x, seg_ts_y):
            dis_state_sf = np.array([0, 0, 0, 0, 0, 0])
            state_sf = np.array([0, 0, 0, 0, 0, 0])
        else:
            multiplier = np.array([1, 0, 0, 0, 0, 1])
            dis_state_sf = (
                (gamma ** (i - 1))
                * get_state_feature(seg_ts_x, seg_ts_y, env)
                * multiplier
            )
            state_sf = get_state_feature(seg_ts_x, seg_ts_y, env) * multiplier

        if use_extended_SF:
            # find action index
            for a_i, action_ in enumerate(GridWorldEnv.actions):
                if action_[0] == seg[i][0] and action_[1] == seg[i][1]:
                    action_index = a_i

            if env.is_terminal(seg_ts_x, seg_ts_y):
                dis_action_sf = np.zeros(env.height * env.width * 4)
            else:
                dis_action_sf = (gamma ** (i - 1)) * get_action_feature(
                    prev_x, prev_y, action_index, env=env
                )

            dis_state_sf = dis_action_sf

            if env.is_terminal(seg_ts_x, seg_ts_y):
                action_sf = np.zeros(env.height * env.width * 4)
            else:
                action_sf = get_action_feature(prev_x, prev_y, action_index, env=env)

            state_sf = action_sf

        phi_dis += dis_state_sf
        phi += state_sf

        prev_x = seg_ts_x
        prev_y = seg_ts_y
    return phi_dis, phi
