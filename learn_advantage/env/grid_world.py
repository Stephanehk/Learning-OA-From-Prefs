import json
import random
import itertools
import enum

import numpy as np


class Entity(enum.IntEnum):
    BLANK = enum.auto()
    GOAL = enum.auto()
    HOUSE = enum.auto()
    SHEEP = enum.auto()
    COIN = enum.auto()
    ROAD_BLOCK = enum.auto()

    MUD = enum.auto()
    MUD_GOAL = enum.auto()
    MUD_HOUSE = enum.auto()
    MUD_SHEEP = enum.auto()
    MUD_COIN = enum.auto()
    MUD_ROAD_BLOCK = enum.auto()


class GridWorldEnv:
    def __init__(self, board_name, height=10, width=10):
        """
        Creates a GridWorld object with the specified parameters. If board_name is not None, then a specific board configuration is loaded from disk.
        """
        self.prev_reward_function = None

        # Number of reward features. We use one feature for each component of the reward function, [gas, goal, sheep, coin, road block, mud]
        self.feature_size = 6
        # The ground truth reward vector, indicating the weights of each reward component.
        self.reward_array = [-1, 50, -50, 1, -1, -2]
        self.multiplier = np.array([1, 0, 0, 0, 0, 1])

        self.ss = (0, 0)
        self.pos = self.ss

        self.height = height
        self.width = width

        if board_name is not None:
            self.n_starts = 92
            board_fp = "random_MDPs/" + board_name + "_board.json"
            reward_fp = "random_MDPs/" + board_name + "_rewards_function.json"
            with open(board_fp, "r") as j:
                self.board = json.loads(j.read())

            with open(reward_fp, "r") as j:
                self.reward_function = json.loads(j.read())
            self.generate_transition_probs()
        else:
            self.n_starts = 0
            self.board = np.zeros((height, width))
            self.reward_function = None

        self.observation_space = len(self.board) * len(self.board[0])
        self.transition_probs = None

    def row_iter(self):
        # x-coordinate
        return range(len(self.board))

    def column_iter(self):
        # x-coordinate
        return range(len(self.board[0]))

    def positions(self):
        return itertools.product(self.row_iter(), self.column_iter())

    def get_entity(self, state):
        if state == 0:
            return Entity.BLANK
        if state == 1:
            return Entity.GOAL
        if state == 2:
            # house
            return Entity.HOUSE
        if state == 3:
            # sheep
            return Entity.SHEEP
        if state == 4:
            # coin
            return Entity.COIN
        if state == 5:
            # road block
            return Entity.ROAD_BLOCK
        if state == 6:
            # mud area
            return Entity.MUD
        if state == 7:
            # mud area + flag
            return Entity.MUD_GOAL
        if state == 8:
            # mud area + house
            return Entity.MUD_HOUSE
        if state == 9:
            # mud area + sheep
            return Entity.MUD_SHEEP
        if state == 10:
            # mud area + coin
            return Entity.MUD_COIN
        if state == 11:
            # mud area + roadblock
            return Entity.MUD_ROAD_BLOCK

    def generate_transition_probs(self):
        """
        This function generates the transition dynamics of the MDP, which by default are deterministic.
        """
        probs = []
        for x in self.row_iter():
            width_nexts = []
            for y in self.column_iter():
                action_nexts = []
                for a_index in range(len(self.actions)):
                    next_state, _, done, reward_feature = self.get_next_state(
                        (x, y), a_index
                    )
                    action_nexts.append(
                        {
                            (
                                next_state,
                                np.dot(reward_feature, self.reward_array),
                                done,
                                tuple(reward_feature),
                            ): 1
                        }
                    )
                width_nexts.append(action_nexts)
            probs.append(width_nexts)
        self.transition_probs = probs

    def set_start_state(self, ss):
        """
        Sets the start state of the agent in the given MDP.

        Input
        - ss: a tuple storing the (x,y) coordinates of the desired start state.
        """
        self.ss = ss
        self.pos = ss

    def find_n_starts(self):
        """
        Finds the number of possible start states. This is the set of all non-terminal and non-blocking states.
        """
        self.n_starts = 0

        for x, y in self.positions():
            if not self.is_terminal(x, y) and not self.is_blocked(x, y):
                self.n_starts += 1

    def get_goal_rand(self):
        """
        Gets a list of goal states.
        """
        self.goals = []
        for x, y in self.positions():
            if self.is_goal(x, y):
                self.goals.append((x, y))
        return random.choice(self.goals)

    def set_custom_reward_function(self, reward_arr, set_global=True):
        """
        Changes the ground truth reward function of the MDP.

        Input
        - reward_arr: a list containing the desired weights for each reward feature.
        - set_global: if true, sets MDPs reward function using reward_arr. If false, does nothing (useful for testing)
        """
        # [gas, goal, sheep, coin, roadblock, mud]
        reward_function = [
            [[0 for a in range(len(self.actions))] for x in range(len(self.board[0]))]
            for y in range(len(self.board))
        ]
        for x, y in self.positions():
            for a_i, a in enumerate(self.actions):
                state = [x, y]
                next_state = [x + a[0], y + a[1]]

                if self.is_terminal(x, y):
                    # means current state is terminal
                    reward_function[x][y][a_i] = 0
                    continue

                if (
                    next_state[0] < 0
                    or next_state[1] < 0
                    or next_state[0] >= self.height
                    or next_state[1] >= self.width
                ):
                    # invalid action
                    if self.board[state[0]][state[1]] < 6:
                        reward_function[x][y][a_i] = reward_arr[0]
                    else:
                        reward_function[x][y][a_i] = reward_arr[5]
                    continue

                obj = self.get_entity(self.board[next_state[0]][next_state[1]])

                if obj == Entity.BLANK:
                    reward_function[x][y][a_i] = reward_arr[0]  # blank
                elif obj == Entity.GOAL:
                    reward_function[x][y][a_i] = reward_arr[1]  # goal
                elif obj == Entity.HOUSE:
                    if self.board[x][y] < 6:
                        reward_function[x][y][a_i] = reward_arr[0]  # blocking state
                    else:
                        reward_function[x][y][a_i] = reward_arr[5]
                elif obj == Entity.SHEEP:
                    reward_function[x][y][a_i] = reward_arr[2]  # sheap
                elif obj == Entity.COIN:
                    reward_function[x][y][a_i] = reward_arr[3] + reward_arr[0]  # coin
                elif obj == Entity.ROAD_BLOCK:
                    reward_function[x][y][a_i] = (
                        reward_arr[4] + reward_arr[0]
                    )  # roadblock
                elif obj == Entity.MUD:
                    reward_function[x][y][a_i] = reward_arr[5]  # mud
                elif obj == Entity.MUD_GOAL:
                    reward_function[x][y][a_i] = reward_arr[1]  # goal
                elif obj == Entity.MUD_HOUSE:
                    if self.board[x][y] < 6:
                        reward_function[x][y][a_i] = reward_arr[
                            0
                        ]  # blocking state + mud
                    else:
                        reward_function[x][y][a_i] = reward_arr[5]
                elif obj == Entity.MUD_SHEEP:
                    reward_function[x][y][a_i] = reward_arr[2]  # sheep
                elif obj == Entity.MUD_COIN:
                    reward_function[x][y][a_i] = (
                        reward_arr[3] + reward_arr[5]
                    )  # coin + mud
                elif obj == Entity.MUD_ROAD_BLOCK:
                    reward_function[x][y][a_i] = (
                        reward_arr[4] + reward_arr[5]
                    )  # roadblock + mud
                else:
                    raise ValueError(
                        f"Unknown state {self.board[next_state[0]][next_state[1]]}"
                    )

        if set_global:
            self.prev_reward_function = self.reward_function
            self.reward_function = reward_function
            self.reward_array = reward_arr
            self.generate_transition_probs()
        return reward_function

    def state2tab(self, x, y):
        """
        Given coordinates in the MDP, converts them to a one-hot vector.
        """
        N = x + len(self.board[0]) * y
        ones = np.zeros(self.observation_space)
        ones[N] = 1
        return ones, N

    def is_blocked(self, x, y):
        """
        Determines if the inputted coordinates are in a blocking state or not.
        """
        return self.board[x][y] in {2, 8}

    def is_terminal(self, x, y):
        """
        Determines if the inputted coordinates are in a terminal state or not.
        """
        return self.board[x][y] in {3, 1, 7, 9}

    def is_goal(self, x, y):
        """
        Determines if the inputted coordinates are a goal state or not.
        """
        return self.board[x][y] in {1, 7}

    def is_valid_move(self, x, y, a):
        """
        Given a set of coordinates and an action, determines if an action is valid. If an action attempts to move an agent outside the bounds of the board
        or into a blocking state it is invalid.

        Input:
        - x,y: the inputted coordinates.
        - a: the inputted action, represented as the array [x displacement, y displacement]

        Output:
        - true if the action is valid, false otherwise.
        """
        return (
            x + a[0] >= 0
            and x + a[0] < len(self.board)
            and y + a[1] >= 0
            and y + a[1] < len(self.board[0])
            and self.board[x + a[0]][y + a[1]] not in {2, 8}
        )

    def get_reward_feature(self, x, y, prev_x, prev_y):
        """
        Returns the reward features for a given transition.

        Input:
        - x,y: the inputted coordinates.
        - prev_x,prev_y: the previous coordinates.

        Output:
        - A list of reward features for the given transition.

        """
        reward_feature = np.zeros(self.feature_size)
        obj = self.get_entity(self.board[x][y])
        if obj == Entity.BLANK:
            reward_feature[0] = 1
        elif obj == Entity.GOAL:
            # flag
            reward_feature[1] = 1
        elif obj == Entity.HOUSE:
            # house
            pass
        elif obj == Entity.SHEEP:
            # sheep
            reward_feature[2] = 1
        elif obj == Entity.COIN:
            # coin
            reward_feature[0] = 1
            reward_feature[3] = 1
        elif obj == Entity.ROAD_BLOCK:
            # road block
            reward_feature[0] = 1
            reward_feature[4] = 1
        elif obj == Entity.MUD:
            # mud area
            reward_feature[5] = 1
        elif obj == Entity.MUD_GOAL:
            # mud area + flag
            reward_feature[1] = 1
        elif obj == Entity.MUD_COIN:
            # mud area + house
            pass
        elif obj == Entity.MUD_SHEEP:
            # mud area + sheep
            reward_feature[2] = 1
        elif obj == Entity.MUD_COIN:
            # mud area + coin
            reward_feature[5] = 1
            reward_feature[3] = 1
        elif obj == Entity.MUD_ROAD_BLOCK:
            # mud area + roadblock
            reward_feature[5] = 1
            reward_feature[4] = 1

        if (x, y) == (prev_x, prev_y):
            reward_feature *= [1, 0, 0, 0, 0, 1]

        return reward_feature

    def get_next_state_prob(self, s, a_index):
        """
        Given a state and the previous action index, returns the next state info for an MDP with stochastic transition dynamics.

        Input:
        - s: the inputted coordinates represented as the tuple (x,y)
        - a_index: the current action index.

        Output:
        - next_state: The next state represented as the tuple (x,y), sampled from the MDP next state distribution.
        - reward: The collected reward.
        - done: If the current transition was into a terminal state or not.
        - reward_feature: The reward feature for the current transition.

        """

        x, y = s
        done = False

        if (
            self.board[x][y] == 3
            or self.board[x][y] == 1
            or self.board[x][y] == 7
            or self.board[x][y] == 9
        ):
            done = True

        transitions = self.transition_probs[x][y][a_index]
        trans = random.choices(
            list(transitions.keys()), weights=transitions.values(), k=1
        )
        next_state, reward, done, phi = trans[0]

        if len(self.reward_array) > 6:
            # means we are using extended SF
            action_phi = np.zeros((self.height, self.width, 4))
            action_phi[x][y][a_index] = 1
            action_phi = np.ravel(action_phi)
            reward += np.dot(action_phi, self.reward_array[6:])

        return next_state, reward, done, list(phi)

    def found_coin(self, phi):
        """
        Given a state feature, returns true if the state corrosponds to a state containing a coin.

        Input:
        - phi: the inputted state feature
        """
        if phi[3] == 1:
            return True
        return False

    def found_goal(self, phi):
        """
        Given a state feature, returns true if the state corrosponds to a state containing the goal.

        Input:
        - phi: the inputted state feature
        """
        if phi[1] == 1:
            return True
        return False

    def get_next_state(self, s, a_index):
        """
        Given a state and the previous action index, returns the next state info for an MDP with deterministic transition dynamics.

        Input:
        - s: the inputted coordinates represented as the tuple (x,y).
        - a_index: the current action index.

        Output:
        - next_state: The next state represented as the tuple (x,y).
        - reward: The collected reward.
        - done: If the current transition was into a terminal state or not.
        - reward_feature: The reward feature for the current transition.

        """

        x, y = s
        prev_x, prev_y = s
        done = False
        a = self.actions[a_index]

        reward = self.reward_function[x][y][a_index]

        if len(self.reward_array) > 6:
            # means we are using extended SF
            action_phi = np.zeros((self.height, self.width, 4))
            action_phi[x][y][a_index] = 1
            action_phi = np.ravel(action_phi)
            reward += np.dot(action_phi, self.reward_array[6:])

        if self.is_valid_move(x, y, a) and not self.is_terminal(x, y):
            x = x + a[0]
            y = y + a[1]
        next_state = (x, y)
        if (
            self.board[x][y] == 3
            or self.board[x][y] == 1
            or self.board[x][y] == 7
            or self.board[x][y] == 9
        ):
            done = True

        reward_feature = self.get_reward_feature(x, y, prev_x, prev_y)

        return next_state, reward, done, reward_feature

    def step(self, a_index):
        """
        Given an action index, returns the next state info and stores next state as a class variable. This is for deterministic transitions.

        Input:
        - a_index: the current action index.

        Output:
        - next_state: The next state represented as the tuple (x,y).
        - reward: The collected reward.
        - done: If the current transition was into a terminal state or not.
        - reward_feature: The reward feature for the current transition.

        """
        next_state, reward, done, reward_feature = self.get_next_state(
            self.pos, a_index
        )
        self.pos = next_state
        return next_state, reward, done, reward_feature

    def step_prob(self, a_index):
        """
        Given an action index, returns the next state info and stores next state as a class variable. This is for stochastic transitions.

        Input:
        - a_index: the current action index.

        Output:
        - next_state: The next state represented as the tuple (x,y), sampled from the next state distribution.
        - reward: The collected reward.
        - done: If the current transition was into a terminal state or not.
        - reward_feature: The reward feature for the current transition.

        """
        next_state, reward, done, reward_feature = self.get_next_state_prob(
            self.pos, a_index
        )
        self.pos = next_state
        return next_state, reward, done, reward_feature

    def reset(self):
        """
        Resets all class variables. This should be called at the end of every episode.
        """

        x = random.randrange(0, self.height)
        y = random.randrange(0, self.width)
        while self.is_blocked(x, y) or self.is_terminal(x, y):
            x = random.randrange(0, self.height)
            y = random.randrange(0, self.width)

        self.ss = (x, y)
        self.pos = self.ss
        return self.pos

    # A list of all possible actions (ie: the 4 cardinal directions.
    actions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    @classmethod
    def find_action_index(cls, action):
        """
        Finds the index of an action represented as an array.

        Input:
        - The specified action, represented as the array [x displacement, y displacement]

        Output:
        - The action index, or false if the action does not exist.
        """
        return cls.actions.index(tuple(action))
