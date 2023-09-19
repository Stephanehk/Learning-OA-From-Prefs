import random
import pickle
import itertools
from itertools import combinations
import math

import numpy as np

from learn_advantage.env.grid_world import GridWorldEnv
from learn_advantage.algorithms.rl_algos import value_iteration
from learn_advantage.env.generate_random_policies import generate_all_policies
from learn_advantage.utils.utils import remove_gt_succ_feat


def randomly_place_item_exact(env, item_id, N, height, width):
    # randomly place mud
    for _ in range(N):
        x = random.randint(0, height - 1)
        y = random.randint(0, width - 1)
        while env.board[x][y] != 0 and env.board[x][y] != 6:
            x = random.randint(0, height - 1)
            y = random.randint(0, width - 1)

        env.board[x][y] = item_id


def is_in_gated_area(x, y, board):
    """
    Checks if coordinates are in the brick area
    """
    return board[x][y] >= 6


def is_in_blocked_area(x, y, board):
    """
    Checks if coordinates are in a blocked area (ie: a house)
    """
    return board[x][y] in {2, 8}


def find_end_state(traj, board):
    """
    Returns the end state of the given segment
    """
    in_gated = False
    traj_ts_x = traj[0][0]
    traj_ts_y = traj[0][1]
    if is_in_gated_area(traj_ts_x, traj_ts_y, board):
        in_gated = True

    for i in range(1, 4):
        if (
            traj_ts_x + traj[i][0] >= 0
            and traj_ts_x + traj[i][0] < len(board)
            and traj_ts_y + traj[i][1] >= 0
            and traj_ts_y + traj[i][1] < len(board[0])
        ):
            if not is_in_blocked_area(
                traj_ts_x + traj[i][0], traj_ts_y + traj[i][1], board
            ):
                next_in_gated = is_in_gated_area(
                    traj_ts_x + traj[i][0], traj_ts_y + traj[i][1], board
                )
                if not in_gated or (in_gated and next_in_gated):
                    traj_ts_x += traj[i][0]
                    traj_ts_y += traj[i][1]
    return traj_ts_x, traj_ts_y


def get_state_feature(env, x, y):
    """
    Returns the reward feature for the given coordinate
    """
    reward_feature = np.zeros(6)
    if env.board[x][y] == 0:
        reward_feature[0] = 1
    elif env.board[x][y] == 1:
        # flag
        reward_feature[1] = 1
    elif env.board[x][y] == 2:
        # house
        pass
    elif env.board[x][y] == 3:
        # sheep
        reward_feature[2] = 1
    elif env.board[x][y] == 4:
        # coin
        reward_feature[0] = 1
        reward_feature[3] = 1
    elif env.board[x][y] == 5:
        # road block
        reward_feature[0] = 1
        reward_feature[4] = 1
    elif env.board[x][y] == 6:
        # mud area
        reward_feature[5] = 1
    elif env.board[x][y] == 7:
        # mud area + flag
        reward_feature[1] = 1
    elif env.board[x][y] == 8:
        # mud area + house
        pass
    elif env.board[x][y] == 9:
        # mud area + sheep
        reward_feature[2] = 1
    elif env.board[x][y] == 10:
        # mud area + coin
        reward_feature[5] = 1
        reward_feature[3] = 1
    elif env.board[x][y] == 11:
        # mud area + roadblock
        reward_feature[5] = 1
        reward_feature[4] = 1
    return reward_feature


def find_reward_features(traj, env, traj_length=3):
    """
    Returns the reward features for the given segment
    """
    gamma = 1
    traj_ts_x = traj[0][0]
    traj_ts_y = traj[0][1]

    prev_x = traj_ts_x
    prev_y = traj_ts_y

    phi = np.zeros(6)
    phi_dis = np.zeros(6)

    for i in range(1, traj_length + 1):
        # check if we are at terminal state
        if (
            env.board[prev_x, prev_y] == 1
            or env.board[traj_ts_x, traj_ts_y] == 3
            or env.board[traj_ts_x, traj_ts_y] == 7
            or env.board[traj_ts_x, traj_ts_y] == 9
        ):
            continue
        if (
            traj_ts_x + traj[i][0] >= 0
            and traj_ts_x + traj[i][0] < len(env.board)
            and traj_ts_y + traj[i][1] >= 0
            and traj_ts_y + traj[i][1] < len(env.board[0])
            and not is_in_blocked_area(
                traj_ts_x + traj[i][0], traj_ts_y + traj[i][1], env.board
            )
        ):
            traj_ts_x += traj[i][0]
            traj_ts_y += traj[i][1]
        if (traj_ts_x, traj_ts_y) != (prev_x, prev_y):
            dis_state_sf = (gamma ** (i - 1)) * get_state_feature(
                env, traj_ts_x, traj_ts_y
            )
            state_sf = get_state_feature(env, traj_ts_x, traj_ts_y)
        # check if we are at terminal state
        elif (
            env.board[prev_x, prev_y] == 1
            or env.board[traj_ts_x, traj_ts_y] == 3
            or env.board[traj_ts_x, traj_ts_y] == 7
            or env.board[traj_ts_x, traj_ts_y] == 9
        ):
            dis_state_sf = [0, 0, 0, 0, 0, 0]
            state_sf = [0, 0, 0, 0, 0, 0]
        else:
            multiplier = np.array([1, 0, 0, 0, 0, 1])
            dis_state_sf = (gamma ** (i - 1)) * (
                get_state_feature(env, traj_ts_x, traj_ts_y) * multiplier
            )
            state_sf = get_state_feature(env, traj_ts_x, traj_ts_y) * multiplier
        phi_dis += dis_state_sf
        phi += state_sf

        prev_x = traj_ts_x
        prev_y = traj_ts_y
    return phi_dis, phi


def create_traj_prob(s0_x, s0_y, action_seq, traj_length, env, values):
    """
    Creates a segment with stochastic transitions
    """
    traj = [(s0_x, s0_y)]
    states = [(s0_x, s0_y)]
    x, y = (s0_x, s0_y)
    t_partial_r_sum = 0
    reward_feature = np.zeros(env.feature_size)
    for i, action in enumerate(action_seq):
        traj.append(action)
        a_i = GridWorldEnv.find_action_index(action)
        next_state, reward, done, phi = env.get_next_state_prob((x, y), a_i)

        assert reward == np.dot(phi, env.reward_array)

        reward_feature += phi

        states.append(next_state)

        if i < traj_length:
            x, y = next_state

        t_partial_r_sum += reward
        is_terminal = done

    assert t_partial_r_sum == np.dot(reward_feature, env.reward_array)
    return (
        traj,
        t_partial_r_sum,
        values[x][y],
        is_terminal,
        x,
        y,
        states,
        reward_feature,
    )


def create_traj(
    s0_x,
    s0_y,
    action_seq,
    traj_length,
    env,
    values,
):
    """
    Creates a segment with deterministic transitions
    """
    # generate trajectory 1
    t_partial_r_sum = 0

    traj = [(s0_x, s0_y)]
    x = s0_x
    y = s0_y
    step_n = 0
    is_terminal = False
    states = [(s0_x, s0_y)]
    for step1 in range(traj_length + 1):
        if env.is_terminal(x, y):
            is_terminal = True

        if step_n != traj_length:
            a = action_seq[step1]

            traj.append(a)
            a_i = GridWorldEnv.find_action_index(a)
            t_partial_r_sum += env.reward_function[x][y][a_i]
            if (
                (
                    x + a[0] >= 0
                    and x + a[0] < len(env.board)
                    and y + a[1] >= 0
                    and y + a[1] < len(env.board[0])
                )
                and not env.is_blocked(x + a[0], y + a[1])
                and not env.is_terminal(x + a[0], y + a[1])
            ):
                x = x + a[0]
                y = y + a[1]
            states.append((x, y))
        step_n += 1

    return traj, t_partial_r_sum, values[x][y], is_terminal, x, y, states


def check_same_policy(prev_Q, curr_Q, env):
    """
    Check if two Q functions will induce the same poilcy
    """
    for x in range(env.height):
        for y in range(env.width):
            if np.argmax(prev_Q[x][y]) != np.argmax(curr_Q[x][y]):
                return False
    return True


def generate_stoch_MDP(r_win):
    """
    Generate an MDP with stochastic transitions
    """
    height = 5
    width = 5

    env = GridWorldEnv(None, height, width)

    # place sheep
    sheep_props = [0.05, 0.1, 0.3]
    sheep_prop = random.choice(sheep_props)
    n_sheep = int(sheep_prop * height * width)
    randomly_place_item_exact(env, 3, n_sheep, height, width)

    # place goal
    x = random.randint(0, height - 1)
    y = random.randint(0, width - 1)
    env.board[x][y] = 1

    rew_vec = [-1, 50, -50, r_win, 0, 0]
    env.set_custom_reward_function(rew_vec, set_global=True)
    env.find_n_starts()

    sheep_trans_prob = 0.5

    for x, y in env.positions():
        for a_index in range(4):
            _, _, _, phi = env.get_next_state((x, y), a_index)

            orig_trans = list(env.transition_probs[x][y][a_index].keys())
            if len(orig_trans) != 1:
                continue
            next_state, reward, done, reward_feature = orig_trans[0]

            if phi[2] == 1:
                # found sheep
                # assert done==True
                env.transition_probs[x][y][a_index] = {
                    (next_state, reward, done, reward_feature): sheep_trans_prob,
                    (next_state, rew_vec[3], done, tuple([0, 0, 0, 1, 0, 0])): 1
                    - sheep_trans_prob,
                }

    # TODO: Missing traj_length
    all_X, all_r, all_ses, all_trajs, all_states, all_actions = subsample_env_trajs(env)
    all_env_boards = None
    return (
        env,
        all_X,
        all_r,
        all_ses,
        all_trajs,
        all_env_boards,
        all_states,
        all_actions,
    )


def generate_MDP(prob=False, n_length_trajs=False):
    """
    Generate an MDP randomly
    """

    dimensions_width = [5, 6, 10]
    dimensions_height = [3, 6, 10, 15]

    height = random.choice(dimensions_height)
    width = random.choice(dimensions_width)

    env = GridWorldEnv(None, height, width)

    # sheep
    sheep_props = [0, 0.1, 0.3]

    sheep_prop = random.choice(sheep_props)
    n_sheep = int(sheep_prop * height * width)
    randomly_place_item_exact(env, 3, n_sheep, height, width)

    # mildly bad
    mildly_bad_props = [0, 0.1, 0.5, 0.8]

    mildly_bad_prop = random.choice(mildly_bad_props)
    while mildly_bad_prop + sheep_prop >= 1:
        mildly_bad_prop = random.choice(mildly_bad_props)
    n_mildly_bad = int(mildly_bad_prop * height * width)
    randomly_place_item_exact(env, 5, n_mildly_bad, height, width)

    # mildly good
    mildly_good_props = [0, 0.1, 0.2]
    mildly_good_prop = random.choice(mildly_good_props)
    while mildly_bad_prop + sheep_prop + mildly_good_prop >= 1:
        mildly_good_prop = random.choice(mildly_good_props)
    n_mildly_good = int(mildly_good_prop * height * width)
    randomly_place_item_exact(env, 4, n_mildly_good, height, width)

    # goal
    x = random.randint(0, height - 1)
    y = random.randint(0, width - 1)
    env.board[x][y] = 1

    goal_rews = [0, 1, 5, 10, 50]
    sheep_rews = [-5, -10, -50, -100]
    mildly_bad_rews = [-2, -5, -10]
    mildly_good_rews = [1]
    mud_rews = [-1, -2, -3]
    rew_vec = [
        -1,
        random.choice(goal_rews),
        random.choice(sheep_rews),
        random.choice(mildly_good_rews),
        random.choice(mildly_bad_rews),
        random.choice(mud_rews),
    ]
    env.set_custom_reward_function(rew_vec, set_global=True)
    env.find_n_starts()

    V, Qs = value_iteration(rew_vec=np.array(rew_vec), gamma=0.999, env=env)

    if prob:
        V, Qs = value_iteration(
            rew_vec=np.array(env.reward_array), gamma=0.999, env=env, is_set=True
        )
        prev_Qs = Qs.copy()

        # get worst state
        worst_state = None
        worst_val = float("inf")
        for x in range(env.height):
            for y in range(env.width):
                if (
                    V[x][y] < worst_val
                    and not env.is_terminal(x, y)
                    and not env.is_blocked(x, y)
                ):
                    worst_val = V[x][y]
                    worst_state = (x, y)

        # create states that randomly jump to goal
        good_tele_prop = 0.75
        good_tele_prob = 0.9
        n_tele_sas = int(good_tele_prop * env.height * env.width * 4)
        sa_teles = []

        # choose random (s,a) pairs to add teleportation
        for _ in range(n_tele_sas):
            x = random.randint(0, env.height - 1)
            y = random.randint(0, env.width - 1)
            a = random.randint(0, 3)
            while (x, y, a) in sa_teles:
                x = random.randint(0, env.height - 1)
                y = random.randint(0, env.width - 1)
                a = random.randint(0, 3)
            sa_teles.append((x, y, a))
        # change selected (s,a) pairs and ensure that the optimal policy has not been changed
        n_failed = 0
        for sa in sa_teles:
            x, y, a = sa
            if np.argmax(Qs[x][y]) == a:
                n_failed += 1
                continue

            # TODO: ASSUMES THAT TRANSITION PROB FOR (S,A) IS DETERMINISTIC AND HAS NOT BEEN MODIFIED YET
            orig_trans = list(env.transition_probs[x][y][a].keys())
            assert len(orig_trans) == 1
            next_state, reward, done, reward_feature = orig_trans[0]
            env.transition_probs[x][y][a] = {
                (next_state, reward, done, reward_feature): good_tele_prob,
                (env.get_goal_rand(), rew_vec[1], True, tuple([0, 1, 0, 0, 0, 0])): 1
                - good_tele_prob,
            }

            V, Qs = value_iteration(
                rew_vec=np.array(env.reward_array), gamma=0.999, env=env, is_set=True
            )

            changed_opt_policy = check_same_policy(prev_Qs, Qs, env)
            if changed_opt_policy:
                env.transition_probs[x][y][a] = {
                    (next_state, reward, done, reward_feature): 1
                }
                n_failed += 1

        coin_trans_prob = 0.95
        goal_trans_prob = 0.99
        # create prob transitions for coins and goal
        for x, y in env.positions():
            for a_index in range(4):
                _, _, _, phi = env.get_next_state((x, y), a_index)

                orig_trans = list(env.transition_probs[x][y][a_index].keys())
                if len(orig_trans) != 1:
                    continue
                next_state, reward, done, reward_feature = orig_trans[0]

                if env.found_coin(phi):
                    # found a coin
                    env.transition_probs[x][y][a_index] = {
                        (next_state, reward, done, reward_feature): coin_trans_prob,
                        (
                            next_state,
                            2 * rew_vec[4] + rew_vec[0],
                            done,
                            tuple([1, 0, 0, 0, 2, 0]),
                        ): 1
                        - coin_trans_prob,
                    }
                if env.found_goal(phi):
                    # found goal
                    # assert done==True
                    env.transition_probs[x][y][a_index] = {
                        (next_state, reward, done, reward_feature): goal_trans_prob,
                        (
                            worst_state,
                            rew_vec[0],
                            False,
                            tuple([1, 0, 0, 0, 0, 0]),
                        ): 1
                        - goal_trans_prob,
                    }

        V, Qs = value_iteration(
            rew_vec=np.array(env.reward_array), gamma=0.999, env=env, is_set=True
        )
        changed_opt_policy = check_same_policy(prev_Qs, Qs, env)
        if changed_opt_policy:
            raise ValueError("Optimal policy has changed.")

    if prob:
        all_X, all_r, all_ses, all_trajs, _, _ = subsample_env_trajs(env)
    elif n_length_trajs:
        # collect N randomly samples segment pairs
        all_Xs = []
        all_rs = []
        all_sess = []
        all_trajss = []
        for traj_length in (3, 6, 9, 12, 15):
            all_X, all_r, all_ses, all_trajs, _, _ = subsample_env_n_length_trajs(
                env, traj_length=traj_length
            )
            all_Xs.append(all_X)
            all_rs.append(all_r)
            all_sess.append(all_ses)
            all_trajss.append(all_trajs)
        return (
            env,
            all_Xs,
            all_rs,
            all_sess,
            all_trajss,
        )
    else:
        # collect N randomly samples segment pairs
        all_X, all_r, all_ses, all_trajs, _ = get_env_trajs(env)

    return env, all_X, all_r, all_ses, all_trajs


def decode(i):
    k = math.floor((1 + math.sqrt(1 + 8 * i)) / 2)
    return k, i - k * (k - 1) // 2


def rand_pair(n):
    return decode(random.randrange(n * (n - 1) // 2))


def rand_pairs(n, m):
    # https://stackoverflow.com/questions/55244113/python-get-random-unique-n-pairs
    return [decode(i) for i in random.sample(range(n * (n - 1) // 2), m)]


def get_action_indices(actions):
    """
    Given actions find their indices
    """
    indices = []
    for action in actions:
        indices.append(GridWorldEnv.find_action_index(action))
    return indices


def get_traj_states(traj, board, traj_length=3):
    """
    Returns a list of all states visited in a segment
    """
    traj_ts_x = traj[0][0]
    traj_ts_y = traj[0][1]

    states = [[traj_ts_x, traj_ts_y]]
    # if is_in_gated_area(traj_ts_x,traj_ts_y):
    #     in_gated = True

    for i in range(1, traj_length + 1):
        if (
            traj_ts_x + traj[i][0] >= 0
            and traj_ts_x + traj[i][0] < len(board)
            and traj_ts_y + traj[i][1] >= 0
            and traj_ts_y + traj[i][1] < len(board[0])
            and not is_in_blocked_area(
                traj_ts_x + traj[i][0], traj_ts_y + traj[i][1], board
            )
        ):
            if not (
                board[traj_ts_x][traj_ts_y] == 1
                or board[traj_ts_x][traj_ts_y] == 3
                or board[traj_ts_x][traj_ts_y] == 7
                or board[traj_ts_x][traj_ts_y] == 9
            ):
                traj_ts_x += traj[i][0]
                traj_ts_y += traj[i][1]

        states.append([traj_ts_x, traj_ts_y])

    return states


def subsample_env_trajs(env, traj_length=3):
    """
    Subsampes segments in an MDP, used for MDPs with stochastic transitions where exhaustively finding all segments is infeasible
    """
    if traj_length == 3:
        all_action_seqs = list(itertools.product(env.actions, env.actions, env.actions))
    else:
        raise ValueError("Unsupported trajectory length")

    V, _ = value_iteration(
        rew_vec=np.array(env.reward_array), gamma=0.999, env=env, is_set=True
    )
    MAX_SAMPLES = 30000
    n_fails = 0
    n_samples = 0

    collected_traj_data = []

    while n_samples <= MAX_SAMPLES and n_fails < 5 * MAX_SAMPLES:
        for x in range(env.height):
            for y in range(env.width):
                for _ in range(5):
                    action_seq_1 = random.choice(all_action_seqs)
                    t1_s0_x, t1_s0_y = (x, y)
                    (
                        traj1,
                        t1_partial_r_sum,
                        v_t1,
                        _,
                        traj1_ts_x,
                        traj1_ts_y,
                        _,
                        phi1,
                    ) = create_traj_prob(
                        t1_s0_x, t1_s0_y, action_seq_1, traj_length, env, V
                    )

                    traj1_ses = [(traj1[0][0], traj1[0][1]), (traj1_ts_x, traj1_ts_y)]

                    v_dif1 = v_t1 - V[t1_s0_x][t1_s0_y]

                    if (
                        tuple(phi1),
                        tuple(traj1),
                        tuple(traj1_ses),
                        t1_partial_r_sum,
                        v_dif1,
                    ) not in collected_traj_data:
                        collected_traj_data.append(
                            (
                                tuple(phi1),
                                tuple(traj1),
                                tuple(traj1_ses),
                                t1_partial_r_sum,
                                v_dif1,
                            )
                        )
                        n_samples += 1
                    else:
                        n_fails += 1

    if (
        len(collected_traj_data) * 100
        > len(collected_traj_data) * (len(collected_traj_data) - 1) // 2
    ):
        pairs_indices = rand_pairs(
            len(collected_traj_data),
            len(collected_traj_data) * (len(collected_traj_data) - 1) // 2,
        )
    else:
        pairs_indices = rand_pairs(
            len(collected_traj_data), len(collected_traj_data) * 100
        )

    all_X = []
    all_r = []
    all_ses = []
    all_trajs = []
    all_states = []
    all_actions = []

    for pair_i in pairs_indices:
        p1_i, p2_i = pair_i

        phi1, traj1, traj1_ses, t1_partial_r_sum, v_dif1 = collected_traj_data[p1_i]
        phi1 = list(phi1)
        traj1 = list(traj1)
        traj1_ses = list(traj1_ses)

        phi2, traj2, traj2_ses, t2_partial_r_sum, _ = collected_traj_data[p2_i]
        phi2 = list(phi2)
        traj2 = list(traj2)
        traj2_ses = list(traj2_ses)

        all_actions.append(
            [get_action_indices(traj1[1:]), get_action_indices(traj2[1:])]
        )
        all_states.append(
            [
                get_traj_states(traj1, env.board, len(traj1) - 1),
                get_traj_states(traj2, env.board, len(traj2) - 1),
            ]
        )

        all_X.append([phi1, phi2])
        all_r.append([t1_partial_r_sum, t2_partial_r_sum])
        all_ses.append([traj1_ses, traj2_ses])
        all_trajs.append([traj1, traj2])

    if len(all_X) > 30000:
        idx = np.random.choice(np.arange(len(all_X)), 30000, replace=False)
        return (
            np.array(all_X)[idx],
            np.array(all_r)[idx],
            np.array(all_ses)[idx],
            np.array(all_trajs)[idx],
            np.array(all_states)[idx],
            np.array(all_actions)[idx],
        )

    return (
        np.array(all_X),
        np.array(all_r),
        np.array(all_ses),
        np.array(all_trajs),
        np.array(all_states),
        np.array(all_actions),
    )


def subsample_env_n_length_trajs(env, traj_length):
    """
    Subsampes segments of varying length in an MDP, used when we want segments of different lengths and finding all such segments is infeasible
    """
    V, _ = value_iteration(
        rew_vec=np.array(env.reward_array), gamma=0.999, env=env, is_set=True
    )
    MAX_SAMPLES = 30000
    n_fails = 0
    n_samples = 0

    collected_traj_data = []

    while n_samples <= MAX_SAMPLES and n_fails < 5 * MAX_SAMPLES:
        for x, y in env.positions():
            for _ in range(5):
                action_seq_1 = [random.choice(env.actions) for i in range(traj_length)]
                t1_s0_x, t1_s0_y = (x, y)
                (
                    traj1,
                    t1_partial_r_sum,
                    v_t1,
                    _,
                    traj1_ts_x,
                    traj1_ts_y,
                    _,
                    phi1,
                ) = create_traj_prob(
                    t1_s0_x, t1_s0_y, action_seq_1, traj_length, env, V
                )

                traj1_ses = [(traj1[0][0], traj1[0][1]), (traj1_ts_x, traj1_ts_y)]

                v_dif1 = v_t1 - V[t1_s0_x][t1_s0_y]

                traj1_info = (
                    tuple(phi1),
                    tuple(traj1),
                    tuple(traj1_ses),
                    t1_partial_r_sum,
                    v_dif1,
                )
                if (traj1_info) not in collected_traj_data:
                    collected_traj_data.append(traj1_info)
                    n_samples += 1
                else:
                    n_fails += 1

    if (
        len(collected_traj_data) * 100
        > len(collected_traj_data) * (len(collected_traj_data) - 1) // 2
    ):
        pairs_indices = rand_pairs(
            len(collected_traj_data),
            len(collected_traj_data) * (len(collected_traj_data) - 1) // 2,
        )
    else:
        pairs_indices = rand_pairs(
            len(collected_traj_data), len(collected_traj_data) * 100
        )

    all_X = []
    all_r = []
    all_ses = []
    all_trajs = []
    all_states = []
    all_actions = []

    for pair_i in pairs_indices:
        p1_i, p2_i = pair_i

        phi1, traj1, traj1_ses, t1_partial_r_sum, _ = collected_traj_data[p1_i]
        phi1 = list(phi1)
        traj1 = list(traj1)
        traj1_ses = list(traj1_ses)

        phi2, traj2, traj2_ses, t2_partial_r_sum, _ = collected_traj_data[p2_i]
        phi2 = list(phi2)
        traj2 = list(traj2)
        traj2_ses = list(traj2_ses)

        all_actions.append(
            [get_action_indices(traj1[1:]), get_action_indices(traj2[1:])]
        )
        all_states.append(
            [
                get_traj_states(traj1, env.board, len(traj1) - 1),
                get_traj_states(traj2, env.board, len(traj2) - 1),
            ]
        )

        all_X.append([phi1, phi2])
        all_r.append([t1_partial_r_sum, t2_partial_r_sum])
        all_ses.append([traj1_ses, traj2_ses])
        all_trajs.append([traj1, traj2])

    if len(all_X) > 30000:
        idx = np.random.choice(np.arange(len(all_X)), 30000, replace=False)
        return (
            np.array(all_X)[idx],
            np.array(all_r)[idx],
            np.array(all_ses)[idx],
            np.array(all_trajs)[idx],
            np.array(all_states)[idx],
            np.array(all_actions)[idx],
        )
    return (
        np.array(all_X),
        np.array(all_r),
        np.array(all_ses),
        np.array(all_trajs),
        np.array(all_states),
        np.array(all_actions),
    )


def get_env_trajs(env, traj_length=3):
    """
    Exhaustively finds all segments of length traj_length in an MDP
    """
    V, _ = value_iteration(rew_vec=np.array(env.reward_array), gamma=0.999, env=env)
    print("finding start states...")
    start_states = []
    for i, j in env.positions():
        if env.is_terminal(i, j) or env.is_blocked(i, j):
            continue
        for i2 in range(len(env.board)):
            for j2 in range(len(env.board[0])):
                if env.is_terminal(i2, j2) or env.is_blocked(i2, j2):
                    continue
                start_states.append([(i, j), (i2, j2)])

    all_action_seqs = list(
        combinations(itertools.product(env.actions, env.actions, env.actions), 2)
    )

    all_collected_trajs = []
    n_pairs_found = 0

    all_X = []
    all_r = []
    all_ses = []
    all_trajs = []
    all_env_boards = []

    seen_traj_pairs = set()

    for start_state in start_states:
        for action_seqs in all_action_seqs:
            action_seqs = list(action_seqs)

            ss1 = start_state[0]
            t1_s0_x, t1_s0_y = ss1
            action_seq_1 = action_seqs[0]
            (
                traj1,
                t1_partial_r_sum,
                v_t1,
                is_terminal1,
                traj1_ts_x,
                traj1_ts_y,
                _,
            ) = create_traj(t1_s0_x, t1_s0_y, action_seq_1, traj_length, env, V)
            # TODO Fix comparison between float and bool.
            if t1_partial_r_sum == False:
                continue

            ss2 = start_state[1]
            t2_s0_x, t2_s0_y = ss2
            action_seq_2 = action_seqs[1]
            (
                traj2,
                t2_partial_r_sum,
                v_t2,
                is_terminal2,
                traj2_ts_x,
                traj2_ts_y,
                _,
            ) = create_traj(t2_s0_x, t2_s0_y, action_seq_2, traj_length, env, V)
            # TODO Fix comparison between float and bool.
            if t2_partial_r_sum == False:
                continue

            if traj1 == traj2:
                continue

            phi1, _ = find_reward_features(traj1, env, len(traj1) - 1)
            phi2, _ = find_reward_features(traj2, env, len(traj2) - 1)

            t1_partial_r_sum = np.dot(env.reward_array, phi1)
            t2_partial_r_sum = np.dot(env.reward_array, phi2)

            # only keep unique trajectorries
            big_traj_pair_tuple = (
                (traj1[0][0], traj1[0][1]),
                (traj1_ts_x, traj1_ts_y),
                (traj2[0][0], traj2[0][1]),
                (traj2_ts_x, traj2_ts_y),
                tuple(phi1),
                tuple(phi2),
            )
            if big_traj_pair_tuple not in seen_traj_pairs:
                seen_traj_pairs.add(big_traj_pair_tuple)
            else:
                continue

            v_dif1 = v_t1 - V[t1_s0_x][t1_s0_y]
            v_dif2 = v_t2 - V[t2_s0_x][t2_s0_y]
            partial_sum_dif = float(t2_partial_r_sum - t1_partial_r_sum)
            v_dif = float(v_dif2 - v_dif1)

            traj1_ses = [(traj1[0][0], traj1[0][1]), (traj1_ts_x, traj1_ts_y)]
            traj2_ses = [(traj2[0][0], traj2[0][1]), (traj2_ts_x, traj2_ts_y)]

            quad = [traj1, traj2, v_dif, partial_sum_dif, (is_terminal1, is_terminal2)]

            all_collected_trajs.append(quad)

            all_X.append([phi1, phi2])
            all_r.append([t1_partial_r_sum, t2_partial_r_sum])
            all_ses.append([traj1_ses, traj2_ses])
            all_trajs.append([traj1, traj2])
            all_env_boards.append(env.board)

            n_pairs_found += 1
    return all_X, all_r, all_ses, all_trajs, all_env_boards


def generate_fig_8_MDP(rew_vec):
    # generate M_1 and M'_1 from figure 8
    width = 1
    height = 4
    env = GridWorldEnv(None, height, width)
    env.board[0][0] = 1
    env.board[3][0] = 3
    env.set_custom_reward_function(rew_vec, set_global=True)
    env.find_n_starts()

    all_X_, all_r_, all_ses_, all_trajs_, _ = get_env_trajs(env)

    all_trajs = []
    all_ses = []
    all_r = []
    all_X = []

    for i, traj in enumerate(all_trajs_):
        if traj[0][0] == (2, 0) and traj[1][0] == (2, 0):
            all_trajs.append(traj)
            all_ses.append(all_ses_[i])
            all_r.append(all_r_[i])
            all_X.append(all_X_[i])

    print(len(all_X))
    return env, all_X, all_r, all_ses, all_trajs


def main():
    gamma = 0.999
    for trial in range(400, 500):
        print("generating MDP: " + str(trial))
        env, all_X, all_r, all_ses, all_trajs = generate_MDP(n_length_trajs=True)
        gt_rew_vec = env.reward_array.copy()

        succ_feats, sa_succ_feats, _, succ_q_feats = generate_all_policies(
            100, gamma, env, gt_rew_vec
        )
        succ_feats, succ_q_feats, sa_succ_feats = remove_gt_succ_feat(
            succ_feats=succ_feats,
            succ_feats_q=succ_q_feats,
            action_succ_feats=sa_succ_feats,
            gt_rew_vec=gt_rew_vec,
            gamma=gamma,
            env=env,
        )

        with open("random_MDPs/MDP_" + str(trial) + "all_trajss.npy", "wb") as f:
            pickle.dump(all_trajs, f)

        np.save("random_MDPs/MDP_" + str(trial) + "gt_rew_vec.npy", gt_rew_vec)
        with open("random_MDPs/MDP_" + str(trial) + "all_Xs.npy", "wb") as f:
            pickle.dump(all_X, f)

        with open("random_MDPs/MDP_" + str(trial) + "all_rs.npy", "wb") as f:
            pickle.dump(all_r, f)

        with open("random_MDPs/MDP_" + str(trial) + "all_sess.npy", "wb") as f:
            pickle.dump(all_ses, f)

        with open("random_MDPs/MDP_" + str(trial) + "env.pickle", "wb") as file:
            pickle.dump(env, file)

        np.save("random_MDPs/MDP_" + str(trial) + "succ_feats.npy", succ_feats)
        np.save("random_MDPs/MDP_" + str(trial) + "succ_q_feats.npy", succ_q_feats)
        np.save("random_MDPs/MDP_" + str(trial) + "sa_succ_feats.npy", sa_succ_feats)

        with open("random_MDPs/MDP_" + str(trial) + "env.pickle", "wb") as file:
            pickle.dump(env, file)


if __name__ == "__main__":
    main()
