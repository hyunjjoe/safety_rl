# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

import gym.spaces
import numpy as np
import gym
from utils import nearest_real_grid_point, visualize_matrix, q_values_from_q_func, \
    index_to_state, v_from_q


class DoubleIntegratorEnv(gym.Env):

    def __init__(self):

        # state bounds
        self.bounds = np.array([[-1, 1],  # 0 axis = state, 1 axis = low vs high
                                [-4, 4]])
        self.control_bounds = np.array([-1, 1])
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]
        self.time_step = 0.01

        # target bounds
        self.target_low = np.array([-1, -4])
        self.target_high = np.array([1, 4])

        # gym variables
        self.action_space = gym.spaces.Discrete(2)  # bang bang controls
        # multiply by 2 to give buffer since ray will crash if outside box
        self.observation_space = gym.spaces.Box(2 * self.low, 2 * self.high)
        self.viewer = None

        # discretization
        self.grid_cells = None

        # internal state
        self.state = np.zeros(2)

        self.seed_val = 0

        # seed
        np.random.seed(self.seed_val)

    def reset(self, start=None):
        """
        if start is given sets the current state to start else randomly resets the state
        :return: the current state of the environment
        """

        if start is None:
            self.state = np.random.uniform(low=self.low, high=self.high)
        else:
            self.state = start
        return np.copy(self.state)

    def step(self, action):
        """
        :param action: input action
        :return: tuple of (resulting state, l of current state, whether the episode is done, info dictionary)
        """
        # note that l must be computed before the environment steps. see readme for a simple proof
        if self.grid_cells is None:
            r = self.signed_distance(self.state)
        else:
            # provide consistent reward signal. same state always same reward from center of grid cell
            r = self.signed_distance(nearest_real_grid_point(self.grid_cells, self.bounds, self.state))

        # move dynamics forward one step
        x, x_dot = self.state
        x = x + self.time_step * x_dot
        u = self.control_bounds[action]
        x_dot = x_dot + self.time_step * u
        self.state = np.array([x, x_dot])

        # calculate done
        #done = np.any(self.state < self.target_low) or np.any(self.state > self.target_high)
        done = x < self.target_low[0] or x > self.target_high[0]
        info = {}
        return np.copy(self.state), r, done, info

    def set_seed(self, seed):
        self.seed_val = seed
        np.random.seed(self.seed_val)

    def signed_distance(self, s):
        """
        :param s: state
        :return: the signed distance of the environment at state s to the failure set
        """
        x_in = s[0] < self.target_low[0] or s[0] > self.target_high[0]
        x_dist = min(s[0] - self.target_low[0], self.target_high[0] - s[0])
        if x_in:
            return -1 * x_dist
        return x_dist

    def set_grid_cells(self, grid_cells):
        """

        :param grid_cells:
        :param bounds:
        :return:
        """
        self.grid_cells = grid_cells

    def set_bounds(self, bounds):
        """

        :param grid_cells:
        :param bounds:
        :return:
        """
        self.bounds = bounds

        # slice low and high from bounds
        self.low = np.array(self.bounds)[:, 0]
        self.high = np.array(self.bounds)[:, 1]

        # Double the range in each state dimension for Gym interface.
        midpoint = (self.low + self.high)/2.0
        interval = self.high - self.low
        self.observation_space = gym.spaces.Box(midpoint - interval,
                                                midpoint + interval)

    def render(self, mode='human'):
        pass

    def ground_truth_comparison(self, q_func):
        """
        compares value function induced from q_func against ground truth for double integrator
        :param q_func: a function that takes in the state and outputs the q values for each action
        at that state
        :return: tuple of (misclassified_safe, misclassified_unsafe)
        """
        computed_v = v_from_q(q_values_from_q_func(q_func, self.grid_cells, self.bounds, 2))
        return self.ground_truth_comparison_v(computed_v)

    def ground_truth_comparison_v(self, computed_v):
        """
        compares value function induced from q_func against ground truth for double integrator
        :param computed_v: a function that takes in the state and outputs the q values for each action
        at that state
        :return: tuple of (misclassified_safe, misclassified_unsafe)
        """
        analytic_v = self.analytic_v()
        misclassified_safe = 0
        misclassified_unsafe = 0
        it = np.nditer(analytic_v, flags=['multi_index'])
        while not it.finished:
            if analytic_v[it.multi_index] < 0 < computed_v[it.multi_index]:
                misclassified_safe += 1
            elif computed_v[it.multi_index] < 0 < analytic_v[it.multi_index]:
                misclassified_unsafe += 1
            it.iternext()
        return misclassified_safe, misclassified_unsafe

    def analytic_range(self):
        """
        the safe set for the double integrator is defined by two parabolas and two lines. this
        returns the coordinates for those two parabolas
        :param num_points: length of each of the lists of coordinates for the parabolas
        :return:
        """

        x_low = self.target_low[0]
        x_high = self.target_high[0]
        u_max = self.control_bounds[1]  # assumes that negative of u_max is u_min
        x_dot_num_points = self.grid_cells[1]

        # edge of range
        x_dot_high = (((x_high - x_low) * (2 * u_max)) ** 0.5)
        x_dot_low = -x_dot_high

        # curve for x_dot < 0
        def x_dot_negative(x_dot):
            return x_low + (x_dot ** 2) / (2 * u_max)

        # curve for x_dot > 0
        def x_dot_positive(x_dot):
            return x_high - (x_dot ** 2) / (2 * u_max)

        # linear ranges of x_dot
        x_dot_negative_range = np.arange(start=x_dot_low, stop=0, step=-x_dot_low / x_dot_num_points)
        x_dot_positive_range = np.arange(start=0, stop=x_dot_high, step=x_dot_high / x_dot_num_points)

        # compute x values
        x_negative_range = x_dot_negative(x_dot_negative_range)
        x_positive_range = x_dot_positive(x_dot_positive_range)
        return [x_dot_negative_range, x_negative_range, x_dot_positive_range, x_positive_range]

    def visualize_analytic_comparison(self, v):
        """
        plots the curves defining the edge of the analytic safe set on top of v
        :param analytic: an output of double_integrator_analytic_range
        :param v: value function to compare against analytic
        :return: None
        """
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib import pyplot as plt
        matplotlib.style.use('ggplot')

        # unpack values from analytic
        analytic = self.analytic_range()
        x_dot_negative_range, x_negative_range, x_dot_positive_range, x_positive_range = analytic

        # plot analytic parabolas
        plt.plot(x_dot_positive_range, x_positive_range, color="black")
        plt.plot(x_dot_negative_range, x_negative_range, color="black")

        # plot analytic lines at edge of set
        plt.plot(x_dot_positive_range, np.ones(len(x_dot_positive_range)), color="black")
        plt.plot(x_dot_negative_range, -1 * np.ones(len(x_dot_negative_range)), color="black")

        # visualize v on top of curves
        visualize_matrix(v, self.get_axes(), no_show=False)

    def analytic_v(self):
        v = np.zeros(self.grid_cells)
        it = np.nditer(v, flags=['multi_index'])

        x_low = self.target_low[0]
        x_high = self.target_high[0]
        u_max = self.control_bounds[1]  # assumes that negative of u_max is u_min

        def analytic_function(x, x_dot):
            if x_dot >= 0:
                return min(x - x_low,
                           x_high - x - x_dot ** 2 / (2 * u_max))
            else:
                return min(x_high - x,
                           x - x_dot ** 2 / (2 * u_max) - x_low)

        while not it.finished:
            x, x_dot = index_to_state(self.grid_cells, self.bounds, it.multi_index)
            v[it.multi_index] = analytic_function(x, x_dot)
            it.iternext()
        return v

    def get_axes(self):
        return [np.append(self.bounds[1], self.bounds[0]), ["x dot", "x"]]
