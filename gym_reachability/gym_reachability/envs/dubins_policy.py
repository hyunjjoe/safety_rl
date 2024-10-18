import gym.spaces
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import torch
import math
import random
from .env_utils import plot_arc, plot_circle
from .env_utils import calculate_margin_circle, calculate_margin_rect, calculate_margin_circle_param
from mlp import gcMLP
from scipy.integrate import solve_ivp
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy
from scipy.stats import truncnorm

class DubinsPolicyEnv(gym.Env):
  """A gym environment considering Dubins car dynamics.
  """

  def __init__(
      self, device, mode="RA", doneType="toEnd", sample_inside_obs=False,
      sample_inside_tar=True
  ):
    """Initializes the environment with given arguments.

    Args:ds
        device (str): device type (used in PyTorch).
        mode (str, optional): reinforcement learning type. Defaults to "RA".
        doneType (str, optional): the condition to raise `done flag in
            training. Defaults to "toEnd".
        sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.
        sample_inside_tar (bool, optional): consider sampling the state inside
            the target if True. Defaults to True.
    """
    # Set random seed.
    self.set_seed(0)
    np.random.seed(self.seed_val)

    # State bounds.
    self.bounds = np.array([[-5, 5], [-5, 5], [0, 2*np.pi], [-3, 3]]) #x, y, th, param
    #self.bounds = np.array([[-5, 5], [-5, 5], [-np.pi, np.pi], [-3, 3]]) #x, y, th, param

    self.low = self.bounds[:, 0]
    self.high = self.bounds[:, 1]
    self.sample_inside_obs = sample_inside_obs
    self.sample_inside_tar = sample_inside_tar

    # Gym variables.
    self.action_space = gym.spaces.Discrete(1)
    self.midpoint = (self.low + self.high) / 2.0
    self.interval = self.high - self.low
    self.observation_space = gym.spaces.Box(
        np.float32(self.midpoint - self.interval/2),
        np.float32(self.midpoint + self.interval/2),
    )

    # Constraint set parameters.
    self.constraint_x_y_w_h = np.array([
      [0., 2.75, 1., 4.5],     # Obstacle 1
      [0., -2.75, 1., 4.5],    # Obstacle 2
      [0., -4.75, 10., 0.5],   # Obstacle 3
      [0., 4.75, 10., 0.5],    # Obstacle 4
      [-4.75, 0, 0.5, 10.],    # Obstacle 5
      [4.75, 0., 0.5, 10.]     # Obstacle 6
    ])

    # Internal state.
    self.mode = mode
    self.state = np.zeros(4)
    self.doneType = doneType

    # Dubins car parameters.
    self.time_step = 0.1
    self.speed = 1

    self.model_device = "cpu"
    self.model = gcMLP(input_size=6, hidden_size=128, output_size=1)
    self.model.load_state_dict(torch.load('dubins_policy/policy_final.pth', map_location=self.model_device))
    self.model.to(self.model_device)
    self.model.eval()
    # Visualization params
    self.visual_initial_states = [
        np.array([-3, -3, 0]),
        np.array([-3, 0, 0]),
        np.array([-3, 3, 0]),
        np.array([-3, -1, 0]),
        np.array([-3, 1, 0])
    ]

    # Cost Params
    self.penalty = 1.0
    self.reward = -1.0
    self.costType = "sparse"
    self.device = device
    self.scaling = 1.0

    print(
        "Env: mode-{:s}; doneType-{:s}; sample_inside_obs-{}".format(
            self.mode, self.doneType, self.sample_inside_obs
        )
    )

  # == Reset Functions ==
  def reset(
      self, start=None, theta=None, sample_inside_obs=False,
      sample_inside_tar=True
  ):
    """Resets the state of the environment.

    Args:
        start (np.ndarray, optional): the state to reset the Dubins car to. If
            None, pick the state uniformly at random. Defaults to None.
        theta (float, optional): if provided, set the initial heading angle
            (yaw). Defaults to None.
        sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.
        sample_inside_tar (bool, optional): consider sampling the state inside
            the target if True. Defaults to True.

    Returns:
        np.ndarray: the state that Dubins car has been reset to.
    """
    if start is None:
      x_rnd, y_rnd, theta_rnd, param_rnd = self.sample_random_state(
          sample_inside_obs=sample_inside_obs,
          sample_inside_tar=sample_inside_tar, theta=theta
      )
      self.state = np.array([x_rnd, y_rnd, theta_rnd, param_rnd])
    else:
      self.state = start
    return np.copy(self.state)

  def set_seed(self, seed):
    """Sets the seed for `numpy`, `random`, `PyTorch` packages.

    Args:
        seed (int): seed value.
    """
    self.seed_val = seed
    np.random.seed(self.seed_val)
    torch.manual_seed(self.seed_val)
    torch.cuda.manual_seed(self.seed_val)
    # if you are using multi-GPU.
    torch.cuda.manual_seed_all(self.seed_val)
    random.seed(self.seed_val)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  # == Getting Functions ==
  def get_warmup_examples(self, num_warmup_samples=100):
    """Gets warmup samples.

    Args:
        num_warmup_samples (int, optional): # warmup samples. Defaults to 100.

    Returns:
        np.ndarray: sampled states.
        np.ndarray: the heuristic values, here we used max{ell, g}.
    """
    rv = np.random.uniform(
        low=self.low, high=self.high, size=(num_warmup_samples, 4)
    )
    x_rnd, y_rnd, theta_rnd, param_rnd = rv[:, 0], rv[:, 1], rv[:, 2], rv[:, 3]

    heuristic_v = np.zeros((num_warmup_samples, self.action_space.n))
    states = np.zeros((num_warmup_samples, self.observation_space.shape[0]))

    for i in range(num_warmup_samples):
      x, y, theta, param = x_rnd[i], y_rnd[i], theta_rnd[i], param_rnd[i]
      l_x = self.target_margin(np.array([x, y, param]))
      g_x = self.safety_margin(np.array([x, y]))
      heuristic_v[i, :] = np.maximum(l_x, g_x)
      states[i, :] = x, y, theta, param

    return states, heuristic_v

  def get_value(self, q_func, theta, param, nx=101, ny=101, addBias=False):
    """
    Gets the state values given the Q-network. We fix the heading angle of the
    car to `theta`.

    Args:
        q_func (object): agent's Q-network.
        theta (float): the heading angle of the car.
        nx (int, optional): # points in x-axis. Defaults to 101.
        ny (int, optional): # points in y-axis. Defaults to 101.
        addBias (bool, optional): adding bias to the values or not.
            Defaults to False.

    Returns:
        np.ndarray: values
    """
    v = np.zeros((nx, ny))
    it = np.nditer(v, flags=["multi_index"])
    xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
    ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
    while not it.finished:
      idx = it.multi_index
      x = xs[idx[0]]
      y = ys[idx[1]]
      l_x = self.target_margin(np.array([x, y, param]))
      g_x = self.safety_margin(np.array([x, y]))

      if self.mode == "normal" or self.mode == "RA":
        state = (torch.FloatTensor([x, y, theta, param]).to(self.device).unsqueeze(0))
      else:
        z = max([l_x, g_x])
        state = (
            torch.FloatTensor([x, y, theta, z]).to(self.device).unsqueeze(0)
        )
      if addBias:
        v[idx] = q_func(state).min(dim=1)[0].item() + max(l_x, g_x)
      else:
        v[idx] = q_func(state).min(dim=1)[0].item()
      it.iternext()
    return v

  def get_comp_value(self, q_func, nx=100, ny=100, nt=50, nparam=20, addBias=False):
      """
      Gets the state values given the Q-network. We fix the heading angle of the
      car to `theta`.

      Args:
          q_func (object): agent's Q-network.
          theta (float): the heading angle of the car.
          nx (int, optional): # points in x-axis. Defaults to 101.
          ny (int, optional): # points in y-axis. Defaults to 101.
          addBias (bool, optional): adding bias to the values or not.
              Defaults to False.

      Returns:
          np.ndarray: values
      """

      #50 50 50 20
      v = np.zeros((nx, ny, nt, nparam))
      it = np.nditer(v, flags=["multi_index"])
      xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
      ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
      ts = np.linspace(self.bounds[2, 0], self.bounds[2, 1], nt)
      ps = np.linspace(self.bounds[3, 0], self.bounds[3, 1], nparam)
      while not it.finished:
        idx = it.multi_index
        x = xs[idx[0]]
        y = ys[idx[1]]
        th = ts[idx[2]]
        param = ps[idx[3]]
        l_x = self.target_margin(np.array([x, y, th, param]))
        g_x = self.safety_margin(np.array([x, y]))

        if self.mode == "normal" or self.mode == "RA":
          state = (torch.FloatTensor([x, y, th, param]).to(self.device).unsqueeze(0))
        else:
          z = max([l_x, g_x])
          state = (
              torch.FloatTensor([x, y, th, z]).to(self.device).unsqueeze(0)
          )
        if addBias:
          v[idx] = q_func(state).min(dim=1)[0].item() + max(l_x, g_x)
        else:
          v[idx] = q_func(state).min(dim=1)[0].item()
        it.iternext()
      return v

  def sample_random_state(
      self, sample_inside_obs=False, sample_inside_tar=True, theta=None
  ):
    """Picks the state uniformly at random.

    Args:
        sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.
        sample_inside_tar (bool, optional): consider sampling the state inside
            the target if True. Defaults to True.
        theta (float, optional): if provided, set the initial heading angle
            (yaw). Defaults to None.

    Returns:
        np.ndarray: the sampled initial state.
    """
    np.random.seed(None)
    # random sample `theta`
    if theta is None:
      theta_rnd = 2.0 * np.random.uniform() * np.pi
      #theta_rnd = np.random.uniform(-np.pi, np.pi)
    else:
      theta_rnd = theta

    # random sample [`x`, `y`]
    flag = True
    while flag:
      rnd_state_param = np.random.uniform(low=self.low, high=self.high)
      l_x = self.target_margin(rnd_state_param)
      g_x = self.safety_margin(rnd_state_param)

      if (not sample_inside_obs) and (g_x > 0):
        flag = True
      elif (not sample_inside_tar) and (l_x <= 0):
        flag = True
      else:
        flag = False
    x_rnd, y_rnd, _, param_rnd = rnd_state_param

    return x_rnd, y_rnd, theta_rnd, param_rnd

 # == Dynamics Functions ==
  def step(self, action):
    """Evolves the environment one step forward given an action.

    Args:
        action (int): the index of the action in the action set.

    Returns:
        np.ndarray: next state.
        float: the standard cost used in reinforcement learning.
        bool: True if the episode is terminated.
        dict: consist of target margin and safety margin at the new state.
    """
    #call the policy
    state_tensor = torch.tensor(np.array([self.state[0], self.state[1], math.sin(self.state[2]), math.cos(self.state[2])]), device=self.model_device, dtype=torch.float32).unsqueeze(0)
    goal_tensor = torch.tensor(np.array([3, self.state[3]]), device=self.model_device, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = self.model(state_tensor, goal_tensor).squeeze().cpu().numpy()
        action = 2 if action > 2 else (-2 if action < -2 else action)
    u = action

    self.state = self.integrate_forward(self.state, u)
    l_x = self.target_margin(self.state)
    g_x = self.safety_margin(self.state[:2])

    fail = g_x > 0
    success = l_x <= 0

    # cost
    if self.mode == "RA":
      if fail:
        cost = self.penalty * self.scaling
      elif success:
        cost = self.reward
      else:
        cost = 0.0
    else:
      if fail:
        cost = self.penalty * self.scaling
      elif success:
        cost = self.reward
      else:
        cost = 0.0

    # done
    if self.doneType == 'toEnd':
      done = not self.check_within_bounds(self.state)
    else:
      assert self.doneType == 'TF', 'invalid doneType'
      fail = g_x > 0
      success = l_x <= 0
      done = fail or success

    # = `info`
    if done and self.doneType == "fail":
      info = {"g_x": self.penalty * self.scaling, "l_x": l_x}
    else:
      info = {"g_x": g_x * self.scaling, "l_x": l_x}

    return np.copy(self.state), cost, done, info

  def integrate_forward(self, state, u):
    """Integrates the dynamics forward by one step.

    Args:
        state (np.ndarray): (x, y, yaw).
        u (float): the contol input, angular speed.

    Returns:
        np.ndarray: next state.
    """
    x, y, theta, param = state

    # x = x + self.time_step * self.speed * np.cos(theta)
    # y = y + self.time_step * self.speed * np.sin(theta)
    # theta = np.mod(theta + self.time_step * u, 2 * np.pi)
    # theta = theta + self.time_step*u
    # if theta > np.pi:
    #   theta = theta - 2*np.pi
    # elif theta < -np.pi:
    #   theta = theta + 2*np.pi
    # param = param
    t_span = [0, self.time_step]
    dynamics = lambda t, s: [self.speed * np.cos(s[2]), self.speed * np.sin(s[2]), u]
    result = solve_ivp(dynamics, t_span, [x, y, theta], method='RK45')
    x, y, theta = result.y[:, -1]
    if theta >= 2*np.pi:
      theta = theta - 2*np.pi
    elif theta < 0:
      theta = theta + 2*np.pi
    state = np.array([x, y, theta, param])
    return state

# == Setting Hyper-Parameter Functions ==
  def set_bounds(self, bounds):
    """Sets the boundary of the environment.

    Args:
        bounds (np.ndarray): of the shape (n_dim, 2). Each row is [LB, UB].
    """
    self.bounds = bounds

    # Get lower and upper bounds
    self.low = np.array(self.bounds)[:, 0]
    self.high = np.array(self.bounds)[:, 1]
    
  def set_time_step(self, time_step=.1):
    """Sets the time step for dynamics integration.

    Args:
        time_step (float, optional): time step used in the integrate_forward.
            Defaults to .05.
    """
    self.time_step = time_step

  # == Getting Functions ==
  def check_within_bounds(self, state):
    """Checks if the agent is still in the environment.

    Args:
        state (np.ndarray): the state of the agent.

    Returns:
        bool: False if the agent is not in the environment.
    """
    for dim, bound in enumerate(self.bounds):
      flagLow = state[dim] < bound[0]
      flagHigh = state[dim] > bound[1]
      if flagLow or flagHigh:
        return False
    return True

  def set_speed(self, speed=1):
    """Sets the linear velocity of the car.

    Args:
        speed (float, optional): speed of the car. Defaults to .5.
    """
    self.speed = speed
  
  def set_target(self, x_center, radius):
    """Sets the target set.

    Args:
        center (np.ndarray, optional): center of the target set.
    """
    self.target_radius = radius
    self.x_center = x_center

  # == Getting Margin ==
  def safety_margin(self, s):
    """Computes the margin (e.g. distance) between the state and the failue set.

    Args:
        s (np.ndarray): the state of the agent.

    Returns:
        float: postivive numbers indicate being inside the failure set (safety
            violation).
    """
    g_x_list = []

    # constraint_set_safety_margin
    for _, constraint_set in enumerate(self.constraint_x_y_w_h):
      g_x = calculate_margin_rect(s, constraint_set, negativeInside=False)
      g_x_list.append(g_x)

    # enclosure_safety_margin
    boundary_x_y_w_h = np.append(self.midpoint[:2], self.interval[:2])
    g_x = calculate_margin_rect(s, boundary_x_y_w_h, negativeInside=True)
    g_x_list.append(g_x)

    safety_margin = np.max(np.array(g_x_list))

    return self.scaling * safety_margin

  def target_margin(self, s):
    """Computes the margin (e.g. distance) between the state and the target set.

    Args:
        s (np.ndarray): the state of the agent.

    Returns:
        float: negative numbers indicate reaching the target. If the target set
            is not specified, return None.
    """
    if self.x_center is not None and self.target_radius is not None:
      target_margin = calculate_margin_circle_param(
          s, [self.x_center, self.target_radius], negativeInside=True
      )
      return target_margin
    else:
      return None

  def get_constraint_set_boundary(self):
    """Gets the constarint set boundary.

    Returns:
        np.ndarray: of the shape (#constraint, 5, 2). Since we use the box
            constraint in this environment, we need 5 points to plot the box.
            The last axis consists of the (x, y) position.
    """
    num_constarint_set = self.constraint_x_y_w_h.shape[0]
    constraint_set_boundary = np.zeros((num_constarint_set, 5, 2))

    for idx, constraint_set in enumerate(self.constraint_x_y_w_h):
      x, y, w, h = constraint_set
      x_l = x - w/2.0
      x_h = x + w/2.0
      y_l = y - h/2.0
      y_h = y + h/2.0
      constraint_set_boundary[idx, :, 0] = [x_l, x_l, x_h, x_h, x_l]
      constraint_set_boundary[idx, :, 1] = [y_l, y_h, y_h, y_l, y_l]

    return constraint_set_boundary



# == Trajectory Functions ==
  def simulate_one_trajectory(
      self, q_func, T=400, state=None, theta=None, sample_inside_obs=True,
      sample_inside_tar=True, toEnd=False, init_q=False
  ):
    """Simulates the trajectory given the state or randomly initialized.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): the maximum length of the trajectory. Defaults
            to 250.
        state (np.ndarray, optional): if provided, set the initial state to
            its value. Defaults to None.
        theta (float, optional): if provided, set the theta to its value.
            Defaults to None.
        sample_inside_obs (bool, optional): sampling initial states inside
            of the obstacles or not. Defaults to True.
        sample_inside_tar (bool, optional): sampling initial states inside
            of the targets or not. Defaults to True.
        toEnd (bool, optional): simulate the trajectory until the robot
            crosses the boundary or not. Defaults to False.

    Returns:
        np.ndarray: states of the trajectory, of the shape (length, 3).
        int: result.
        float: the minimum reach-avoid value of the trajectory.
        dictionary: extra information, (v_x, g_x, ell_x) along the traj.
    """
    # reset
    if state is None:
      state = self.sample_random_state(
          sample_inside_obs=sample_inside_obs,
          sample_inside_tar=sample_inside_tar,
          theta=theta,
      )
    # while True:
    #   state = self.sample_random_state(
    #       sample_inside_obs=sample_inside_obs,
    #       sample_inside_tar=sample_inside_tar,
    #       theta=theta,
    #   )
    #   state_tensor = (torch.FloatTensor(state).to(self.device).unsqueeze(0))
    #   initial_q = q_func(state_tensor).item()
    #   if(abs(initial_q) < 0.1):
    #     print(initial_q)
    #     break
    traj = []
    result = 0  # not finished
    valueList = []
    gxList = []
    lxList = []
    actions = []
    initial_q = None
    if initial_q is None:
      state_tensor = (torch.FloatTensor(state).to(self.device).unsqueeze(0))
      initial_q = q_func(state_tensor).item()
    self.state = state
    for t in range(T):
      traj.append(self.state)

      g_x = self.safety_margin(self.state)
      l_x = self.target_margin(self.state)

      # = Rollout Record
      if t == 0:
        maxG = g_x
        current = max(l_x, maxG)
        minV = current
      else:
        maxG = max(maxG, g_x)
        current = max(l_x, maxG)
        minV = min(current, minV)

      valueList.append(minV)
      gxList.append(g_x)
      lxList.append(l_x)

      if toEnd:
        done = not self.check_within_bounds(self.state)
        if done:
          result = 1
          break
      else:
        if g_x > 0:
          result = -1  # failed
          break
        elif l_x <= 0:
          result = 1  # succeeded
          break

        #call the policy
        #print(self.state)
        state_tensor = torch.tensor(np.array([self.state[0], self.state[1], math.sin(self.state[2]), math.cos(self.state[2])]), device=self.model_device, dtype=torch.float32).unsqueeze(0)
        goal_tensor = torch.tensor(np.array([3, self.state[3]]), device=self.model_device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
          action = self.model(state_tensor, goal_tensor).squeeze().cpu().numpy()
          action = 2 if action > 2 else (-2 if action < -2 else action)
      u = action
      #print(u)
      actions.append(u)
      self.state = self.integrate_forward(self.state, u)
    traj = np.array(traj)
    info = {"valueList": valueList, "gxList": gxList, "lxList": lxList}
    if result ==0:
      result = -1
    if init_q:
      return traj, result, initial_q
    return traj, result, minV, info

  def simulate_trajectories(
      self, q_func, T=250, num_rnd_traj=None, states=None, toEnd=False
  ):
    """
    Simulates the trajectories. If the states are not provided, we pick the
    initial states from the discretized state space.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): the maximum length of the trajectory.
            Defaults to 250.
        num_rnd_traj (int, optional): #trajectories. Defaults to None.
        states (list of np.ndarray, optional): if provided, set the initial
            states to its value. Defaults to None.
        toEnd (bool, optional): simulate the trajectory until the robot
            crosses the boundary or not. Defaults to False.

    Returns:
        list of np.ndarray: each element is a tuple consisting of x and y
            positions along the trajectory.
        np.ndarray: the binary reach-avoid outcomes.
        np.ndarray: the minimum reach-avoid values of the trajectories.
    """
    assert ((num_rnd_traj is None and states is not None)
            or (num_rnd_traj is not None and states is None)
            or (len(states) == num_rnd_traj))
    trajectories = []

    if states is None:
      nx = 11
      ny = nx
      nparam = nx
      xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
      ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
      params = np.linspace(self.bounds[3, 0], self.bounds[3, 1], nparam)
      results = np.empty((nx, ny, nparam), dtype=int)
      minVs = np.empty((nx, ny, nparam), dtype=float)

      it = np.nditer(results, flags=["multi_index"])
      print()
      while not it.finished:
        idx = it.multi_index
        print(idx, end="\r")
        x = xs[idx[0]]
        y = ys[idx[1]]
        param = params[idx[2]]
        state = np.array([x, y, 0, param])
        traj, result, minV, _ = self.simulate_one_trajectory(
            q_func, T=T, state=state, toEnd=toEnd
        )
        trajectories.append((traj))
        results[idx] = result
        minVs[idx] = minV
        it.iternext()
      results = results.reshape(-1)
      minVs = minVs.reshape(-1)

    else:
      results = np.empty(shape=(len(states),), dtype=int)
      minVs = np.empty(shape=(len(states),), dtype=float)
      for idx, state in enumerate(states):
        traj, result, minV, _ = self.simulate_one_trajectory(
            q_func, T=T, state=state, toEnd=toEnd
        )
        trajectories.append(traj)
        results[idx] = result
        minVs[idx] = minV

    return trajectories, results, minVs

# == Plotting Functions ==
  def render(self):
    pass

  def visualize(
      self, q_func, vmin=-1, vmax=1, nx=101, ny=101, cmap="seismic",
      labels=None, boolPlot=False, addBias=False, theta=0,
      rndTraj=False, num_rnd_traj=10
  ):
    """
    Visulaizes the trained Q-network in terms of state values and trajectories
    rollout.

    Args:
        q_func (object): agent's Q-network.
        vmin (int, optional): vmin in colormap. Defaults to -1.
        vmax (int, optional): vmax in colormap. Defaults to 1.
        nx (int, optional): # points in x-axis. Defaults to 101.
        ny (int, optional): # points in y-axis. Defaults to 101.
        cmap (str, optional): color map. Defaults to 'seismic'.
        labels (list, optional): x- and y- labels. Defaults to None.
        boolPlot (bool, optional): plot the values in binary form.
            Defaults to False.
        addBias (bool, optional): adding bias to the values or not.
            Defaults to False.
        theta (float, optional): if provided, set the theta to its value.
            Defaults to np.pi/2.
        rndTraj (bool, optional): randomli choose trajectories if True.
            Defaults to False.
        num_rnd_traj (int, optional): #trajectories. Defaults to None.
    """
    theta= 0 #Default theta to 0 
    paramList = [-3, 0, 3]
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    axList = [ax1, ax2, ax3]

    for i, (ax, param) in enumerate(zip(axList, paramList)):
      # for i, (ax, theta) in enumerate(zip(self.axes, thetaList)):
      ax.cla()
      if i == len(paramList) - 1:
        cbarPlot = True
      else:
        cbarPlot = False

      # == Plot failure / target set ==
      self.plot_target_failure_set(ax, param_loc=param)

      # == Plot reach-avoid set ==
      #self.plot_reach_avoid_set(ax, orientation=theta)

      # == Plot V ==
      self.plot_v_values(
          q_func,
          ax=ax,
          fig=fig,
          theta=theta,
          param = param,
          vmin=vmin,
          vmax=vmax,
          nx=nx,
          ny=ny,
          cmap=cmap,
          boolPlot=boolPlot,
          cbarPlot=cbarPlot,
          addBias=addBias,
      )
      # == Formatting ==
      self.plot_formatting(ax=ax, labels=labels)

      # == Plot Trajectories ==
      if rndTraj:
        self.plot_trajectories(
            q_func,
            T=250,
            num_rnd_traj=num_rnd_traj,
            param = param,
            theta=theta,
            toEnd=False,
            ax=ax,
            c="y",
            lw=2,
            orientation=0,
        )
      else:
        # `visual_initial_states` are specified for theta = pi/2. Thus,
        # we need to use "orientation = theta-pi/2"
        self.plot_trajectories(
            q_func,
            T=250,
            states=self.visual_initial_states,
            param = param,
            toEnd=False,
            ax=ax,
            c="y",
            lw=2,
            orientation=theta,
        )

      ax.set_xlabel(
          r"$param={:.0f}$".format(param),
          fontsize=28,
      )

    plt.tight_layout()

  def plot_v_values(
      self, q_func, theta=0, param = 0, ax=None, fig=None, vmin=-1, vmax=1,
      nx=201, ny=201, cmap="seismic", boolPlot=True, cbarPlot=True,
      addBias=False
  ):
    """Plots state values.

    Args:
        q_func (object): agent's Q-network.
        theta (float, optional): if provided, fix the car's heading angle
            to its value. Defaults to np.pi/2.
        ax (matplotlib.axes.Axes, optional): Defaults to None.
        fig (matplotlib.figure, optional): Defaults to None.
        vmin (int, optional): vmin in colormap. Defaults to -1.
        vmax (int, optional): vmax in colormap. Defaults to 1.
        nx (int, optional): # points in x-axis. Defaults to 201.
        ny (int, optional): # points in y-axis. Defaults to 201.
        cmap (str, optional): color map. Defaults to 'seismic'.
        boolPlot (bool, optional): plot the values in binary form.
            Defaults to False.
        cbarPlot (bool, optional): plot the color bar or not. Defaults to True.
        addBias (bool, optional): adding bias to the values or not.
            Defaults to False.
    """
    axStyle = self.get_axes()
    ax.plot([0.0, 0.0], [axStyle[0][2], axStyle[0][3]], c="k")
    ax.plot([axStyle[0][0], axStyle[0][1]], [0.0, 0.0], c="k")

    # == Plot V ==
    if theta is None:
      #theta = np.random.uniform(-np.pi, np.pi)
      theta = np.random.uniform(0, 2*np.pi)
    v = self.get_value(q_func, theta, param, nx, ny, addBias=addBias)

    if boolPlot:
      im = ax.imshow(
          v.T > 0.0,
          interpolation="none",
          extent=axStyle[0],
          origin="lower",
          cmap=cmap,
          zorder=-1,
      )
    else:
      im = ax.imshow(
          v.T,
          interpolation="none",
          extent=axStyle[0],
          origin="lower",
          cmap=cmap,
          vmin=vmin,
          vmax=vmax,
          zorder=-1,
      )
      if cbarPlot:
        cbar = fig.colorbar(
            im,
            ax=ax,
            pad=0.01,
            fraction=0.05,
            shrink=0.95,
            ticks=[vmin, 0, vmax],
        )
        cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=16)
    ax.set_xlim(self.bounds[0, 0], self.bounds[0, 1])
    ax.set_ylim(self.bounds[1, 0], self.bounds[1, 1])

  def plot_trajectories(
      self, q_func, T=250, num_rnd_traj=None, states=None, theta=None, param = None,
      toEnd=False, ax=None, c="y", lw=1.5, orientation=0, zorder=2
  ):
    """Plots trajectories given the agent's Q-network.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): the maximum length of the trajectory.
            Defaults to 100.
        num_rnd_traj (int, optional): #states. Defaults to None.
        states (list of np.ndarray, optional): if provided, set the initial
            states to its value. Defaults to None.
        theta (float, optional): if provided, set the car's heading angle
            to its value. Defaults to None.
        toEnd (bool, optional): simulate the trajectory until the robot
            crosses the boundary or not. Defaults to False.
        ax (matplotlib.axes.Axes, optional): Defaults to None.
        c (str, optional): color. Defaults to 'y'.
        lw (float, optional): linewidth. Defaults to 1.5.
        orientation (float, optional): counter-clockwise angle. Defaults
            to 0.
        zorder (int, optional): graph layers order. Defaults to 2.

    Returns:
        np.ndarray: the binary reach-avoid outcomes.
        np.ndarray: the minimum reach-avoid values of the trajectories.
    """
    assert ((num_rnd_traj is None and states is not None)
            or (num_rnd_traj is not None and states is None)
            or (len(states) == num_rnd_traj))

    if states is not None:
      tmpStates = []
      for state in states:
        x, y, theta = state
        xtilde = x * np.cos(orientation) - y * np.sin(orientation)
        ytilde = y * np.cos(orientation) + x * np.sin(orientation)
        thetatilde = theta + orientation
        tmpStates.append(np.array([xtilde, ytilde, thetatilde, param]))
      states = tmpStates

    trajectories, results, minVs = self.simulate_trajectories(
        q_func, T=T, num_rnd_traj=num_rnd_traj, states=states, toEnd=toEnd
    )
    if ax is None:
      ax = plt.gca()
    for traj in trajectories:
      traj_x = traj[:, 0]
      traj_y = traj[:, 1]
      ax.scatter(traj_x[0], traj_y[0], s=48, c=c, zorder=zorder)
      ax.plot(traj_x, traj_y, color=c, linewidth=lw, zorder=zorder)

    return results, minVs

  def plot_target_failure_set(self, ax=None, param_loc=3, c_c="m", c_t="y", lw=3, zorder=0):
    """Plots the boundary of the target and the failure set.

    Args:
        ax (matplotlib.axes.Axes, optional): ax to plot.
        c_c (str, optional): color of the constraint set boundary.
            Defaults to 'm'.
        c_t (str, optional): color of the target set boundary.
            Defaults to 'y'.
        lw (float, optional): linewidth of the boundary. Defaults to 3.
        zorder (int, optional): graph layers order. Defaults to 0.
    """
    for one_boundary in self.get_constraint_set_boundary():
      ax.plot(
          one_boundary[:, 0], one_boundary[:, 1], color=c_c, lw=lw,
          zorder=zorder
      )
      
    plot_circle(
        np.array([self.x_center, param_loc]),
        self.target_radius,
        ax,
        c=c_t,
        lw=lw,
        zorder=zorder,
    )
  
  def plot_reach_avoid_set(self, q_func, nx=101, ny=101, theta=0, param=1.5):
      obstacles = [
          [-0.5, 0.5, 0.5, 5],     # Obstacle 1
          [-0.5, 0.5, -5, -0.5],   # Obstacle 2
          [-5, 5, -5, -4.5],       # Obstacle 3
          [-5, 5, 4.5, 5],         # Obstacle 4
          [-5, -4.5, -5, 5],       # Obstacle 5
          [4.5, 5, -5, 5]          # Obstacle 6
      ]
      fig, ax = plt.subplots(figsize=(10, 10))

      # Plot the goals
      goali_circle = patches.Ellipse(np.array([3, param]), width=1, height=1, color='#C8A2C8', fill=False, linewidth=2)
      ax.add_patch(goali_circle)

      axStyle = self.get_axes()
      x = np.linspace(axStyle[0][0], axStyle[0][1], nx)
      y = np.linspace(axStyle[0][2], axStyle[0][3], ny)
      X, Y = np.meshgrid(x, y)
      v = self.get_value(q_func, theta, param, nx, ny)

      # Plot the zero-level contour with a lower z-order
      ax.contour(X, Y, v.T, levels=[0], colors='#C8A2C8', linewidths=2, zorder=1)

      # Plot the obstacles with a higher z-order
      for bounds in obstacles:
          rect = patches.Rectangle((bounds[0], bounds[2]),
                                  bounds[1] - bounds[0],
                                  bounds[3] - bounds[2],
                                  linewidth=1, edgecolor='black', facecolor='black', zorder=2)
          ax.add_patch(rect)

      # Additional plot settings
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_aspect('equal')
      ax.set_xlim(self.bounds[0, 0], self.bounds[0, 1])
      ax.set_ylim(self.bounds[1, 0], self.bounds[1, 1])

      plt.tight_layout()
      plt.subplots_adjust(wspace=0.025)

      # Save the plot
      dir = 'plots'
      if not os.path.exists(dir):
          os.makedirs(dir)
      fig.savefig(os.path.join(dir, 'ra_set.png'), bbox_inches='tight')
      plt.show()

  def plot_v_values_paper(
    self, q_func, theta=0, param=2, vmin=-1, vmax=1,
    nx=201, ny=201, cmap="seismic", addBias=False
):
    """Plots state values.

    Args:
        q_func (object): agent's Q-network.
        theta (float, optional): if provided, fix the car's heading angle
            to its value. Defaults to np.pi/2.
        vmin (int, optional): vmin in colormap. Defaults to -1.
        vmax (int, optional): vmax in colormap. Defaults to 1.
        nx (int, optional): # points in x-axis. Defaults to 201.
        ny (int, optional): # points in y-axis. Defaults to 201.
        cmap (str, optional): color map. Defaults to 'seismic'.
        addBias (bool, optional): adding bias to the values or not.
            Defaults to False.
    """
    obstacles = [
        [-0.5, 0.5, 0.5, 5],     # Obstacle 1
        [-0.5, 0.5, -5, -0.5],   # Obstacle 2
        [-5, 5, -5, -4.5],       # Obstacle 3
        [-5, 5, 4.5, 5],         # Obstacle 4
        [-5, -4.5, -5, 5],       # Obstacle 5
        [4.5, 5, -5, 5]          # Obstacle 6
    ]
    fig, ax = plt.subplots(figsize=(10, 10))
    axStyle = self.get_axes()
    goali_circle = patches.Ellipse(np.array([3, param]), width=1, height=1, color='red', fill=True, linewidth=2)
    ax.add_patch(goali_circle)
    # == Plot V ==
    if theta is None:
        theta = np.random.uniform(0, 2 * np.pi)
    v = self.get_value(q_func, theta, param, nx, ny, addBias=addBias)
    im = ax.imshow(
        v.T,
        interpolation="none",
        extent=axStyle[0],
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=-1,
    )
    cbar = fig.colorbar(
        im,
        ax=ax,
        pad=0.01,
        fraction=0.05,
        shrink=0.95,
        ticks=[vmin, 0, vmax],
    )
    cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=16)
    # Plot the obstacles with a higher z-order
    for bounds in obstacles:
        rect = patches.Rectangle((bounds[0], bounds[2]),
                                 bounds[1] - bounds[0],
                                 bounds[3] - bounds[2],
                                 linewidth=1, edgecolor='black', facecolor='black', zorder=2)
        ax.add_patch(rect)

    # Additional plot settings
    ax.set_aspect('equal')
    ax.set_xlim(self.bounds[0, 0], self.bounds[0, 1])
    ax.set_ylim(self.bounds[1, 0], self.bounds[1, 1])
    
    # Completely turn off the axes
    ax.axis('off')

    # Save the plot
    dir = 'plots'
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.savefig(os.path.join(dir, 'val_set.png'), bbox_inches='tight')
    plt.show()


  def get_axes(self):
    """Gets the axes bounds and aspect_ratio.

    Returns:
        np.ndarray: axes bounds.
        float: aspect ratio.
    """
    aspect_ratio = ((self.bounds[0, 1] - self.bounds[0, 0]) /
                    (self.bounds[1, 1] - self.bounds[1, 0]))
    axes = np.array([
        self.bounds[0, 0],
        self.bounds[0, 1],
        self.bounds[1, 0],
        self.bounds[1, 1],
    ])
    return [axes, aspect_ratio]
  def plot_formatting(self, ax=None, labels=None):
    """Formats the visualization.

    Args:
        ax (matplotlib.axes.Axes, optional): ax to plot.
        labels (list, optional): x- and y- labels. Defaults to None.
    """
    axStyle = self.get_axes()
    # == Formatting ==
    ax.axis(axStyle[0])
    ax.set_aspect(axStyle[1])  # makes equal aspect ratio
    ax.grid(False)
    if labels is not None:
      ax.set_xlabel(labels[0], fontsize=52)
      ax.set_ylabel(labels[1], fontsize=52)

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
    )
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.xaxis.set_major_formatter("{x:.1f}")
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_formatter("{x:.1f}")

  def confusion_matrix(self, q_func, num_states=100):
    """Gets the confusion matrix using DDQN values and rollout results.

    Args:
        q_func (object): agent's Q-network.
        num_states (int, optional): # initial states to rollout a trajectoy.
            Defaults to 50.

    Returns:
        np.ndarray: confusion matrix.
    """
    # nx = 50
    # ny = 50
    # nth = 20
    # nparam = 20
    # num_states = nx*ny*nth*nparam
    # xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
    # ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
    # ths = np.linspace(self.bounds[2, 0], self.bounds[2, 1], ny)
    # params = np.linspace(self.bounds[3, 0], self.bounds[3, 1], nparam)
    # results = np.empty((nx, ny, nth, nparam), dtype=int)
    # minVs = np.empty((nx, ny, nth, nparam), dtype=float)
    # it = np.nditer(results, flags=["multi_index"])
    # print()
    # confusion_matrix = np.array([[0.0, 0.0], [0.0, 0.0]])
    # while not it.finished:
    #   idx = it.multi_index
    #   print(idx, end="\r")
    #   x = xs[idx[0]]
    #   y = ys[idx[1]]
    #   th = ths[idx[2]]
    #   param = params[idx[3]]
    #   state = np.array([x, y, th, param])
    #   _, result, initial_q = self.simulate_one_trajectory(
    #       q_func, state=state, init_q=True
    #   )
    #   assert (result == 1) or (result == -1)
    #   # note that signs are inverted
    #   if -int(np.sign(initial_q)) == np.sign(result):
    #     if np.sign(result) == 1:
    #       # True Positive. (reaches and it predicts so)
    #       confusion_matrix[0, 0] += 1.0
    #     elif np.sign(result) == -1:
    #       # True Negative. (collides and it predicts so)
    #       confusion_matrix[1, 1] += 1.0
    #   else:
    #     if np.sign(result) == 1:
    #       # False Positive.(reaches target, predicts it will collide)
    #       confusion_matrix[0, 1] += 1.0
    #     elif np.sign(result) == -1:
    #       # False Negative.(collides, predicts it will reach target)
    #       confusion_matrix[1, 0] += 1.0
    #   it.iternext()
    # return confusion_matrix / num_states

    confusion_matrix = np.array([[0.0, 0.0], [0.0, 0.0]])
    for ii in range(num_states):
      _, result, initial_q = self.simulate_one_trajectory(
          q_func, init_q=True
      )
      assert (result == 1) or (result == -1)
      # note that signs are inverted
      if -int(np.sign(initial_q)) == np.sign(result):
        if np.sign(result) == 1:
          # True Positive. (reaches and it predicts so)
          confusion_matrix[0, 0] += 1.0
        elif np.sign(result) == -1:
          # True Negative. (collides and it predicts so)
          confusion_matrix[1, 1] += 1.0
      else:
        if np.sign(result) == 1:
          # False Positive.(reaches target, predicts it will collide)
          confusion_matrix[0, 1] += 1.0
        elif np.sign(result) == -1:
          # False Negative.(collides, predicts it will reach target)
          confusion_matrix[1, 0] += 1.0
    return confusion_matrix / num_states
  
  def ooa(self, q_func):
    obstacles = [
      [-0.5, 0.5, 0.5, 5],     # Obstacle 1
      [-0.5, 0.5, -5, -0.5],   # Obstacle 2
      [-5, 5, -5, -4.5],       # Obstacle 3
      [-5, 5, 4.5, 5],         # Obstacle 4
      [-5, -4.5, -5, 5],       # Obstacle 5
      [4.5, 5, -5, 5]          # Obstacle 6
    ]

    loop_continue = True  # Control flag for the outer loop
    state = np.array([-2,-4.2,0,3])
    goal_input = None
    while loop_continue:
        g_y = float(input("Enter goal y_pos: "))
        goal_input = g_y
        state = np.array(state)
        state[-1] = g_y
        # Checking if the target is reachable
        state_tensor = (torch.FloatTensor(state).to(self.device).unsqueeze(0))
        initial_q = q_func(state_tensor).item()
        if initial_q <= 0:
            print("Target Reachable!")
            break
        else:
            # Informed Sampling
            mean = g_y
            var = 0.1  # Starting variance
            lb = self.bounds[3,0]
            ub = self.bounds[3,1]
            while True:
                sd = np.sqrt(var)
                if sd==math.inf:
                    print("No goal can be reached, input a new initial state.")
                    break
                a, b = (lb - mean) / sd, (ub - mean) / sd
                truncated_gaussian = truncnorm(a, b, loc=mean, scale=sd)
                samples = truncated_gaussian.rvs(100)
                found = False
                for sample in samples:
                    state[-1] = sample
                    state_tensor = (torch.FloatTensor(state).to(self.device).unsqueeze(0))
                    initial_q = q_func(state_tensor).item()
                    if initial_q <= 0:
                        g_y = sample
                        found = True
                        break
                if found:
                    ans = input(f"New goal has been proposed at g_y = {g_y}, accept (Y/N)? ")
                    if ans == "Y":
                        print("New goal accepted.")
                        state[-1] = g_y
                        loop_continue = False
                        break
                    elif ans == "N":
                        print("Restarting, please propose a new goal.")
                        break
                var *= 2

    state[-1] = g_y
    traj, result, _, _ = self.simulate_one_trajectory(state=state,q_func=q_func)
    state_input = deepcopy(state)
    state_input[-1] = goal_input
    traj_input, _, _, _ = self.simulate_one_trajectory(state=state_input,q_func=q_func)
    # Visualize trajectory and corresponding BRAT
    # Visualize trajectory and corresponding BRAT in one plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot environment
    for bounds in obstacles:
        rect = patches.Rectangle((bounds[0], bounds[2]),
                                bounds[1] - bounds[0],
                                bounds[3] - bounds[2],
                                linewidth=1, edgecolor='black', facecolor='black')
        ax.add_patch(rect)

    # Plot the goals
    goali_circle = patches.Ellipse(np.array([3, goal_input]), width=1, height=1, color='red', fill=True, linewidth=2)
    goal1_circle = patches.Ellipse(np.array([3, state[-1]]), width=1, height=1, color='green', fill=True, linewidth=2)
    ax.add_patch(goal1_circle)
    ax.add_patch(goali_circle)

    # Plot arrows for trajectories
    arrow_start_idx1 = int(0.9 * len(traj))  # Start the arrow at 90% of the trajectory length
    arrow_end_idx1 = len(traj) - 1  # End the arrow at the last point
    dx1 = traj[arrow_end_idx1, 0] - traj[arrow_start_idx1, 0]
    dy1 = traj[arrow_end_idx1, 1] - traj[arrow_start_idx1, 1]
    arrow1 = patches.FancyArrowPatch((traj[arrow_start_idx1, 0], traj[arrow_start_idx1, 1]),
                                    (traj[arrow_start_idx1, 0] + dx1 * 0.1, traj[arrow_start_idx1, 1] + dy1 * 0.1),
                                    arrowstyle='->', color='green', mutation_scale=40, linewidth=3)
    ax.add_patch(arrow1)

    arrow_start_idx2 = int(0.4 * len(traj_input))  # Start the arrow at 40% of the trajectory length
    arrow_end_idx2 = len(traj_input) - 1  # End the arrow at the last point
    dx2 = traj_input[arrow_end_idx2, 0] - traj_input[arrow_start_idx2, 0]
    dy2 = traj_input[arrow_end_idx2, 1] - traj_input[arrow_start_idx2, 1]
    arrow2 = patches.FancyArrowPatch((traj_input[arrow_start_idx2, 0], traj_input[arrow_start_idx2, 1]),
                                    (traj_input[arrow_start_idx2, 0] + dx2 * 0.1, traj_input[arrow_start_idx2, 1] + dy2 * 0.1),
                                    arrowstyle='->', color='red', mutation_scale=40, linewidth=3)
    ax.add_patch(arrow2)

    # Plot trajectories
    ax.plot(traj[:, 0], traj[:, 1], linewidth=3, color='green')
    ax.plot(traj_input[:, 0], traj_input[:, 1], linewidth=3, color='red')

    # Additional plot settings
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_xlim(self.bounds[0, 0], self.bounds[0, 1])
    ax.set_ylim(self.bounds[1, 0], self.bounds[1, 1])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.025)

    # Save the plot
    dir = 'plots'
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.savefig(os.path.join(dir, 'OOA_rollout.png'), bbox_inches='tight')


  def ooa_test(self, q_func, ic=1000):
    obstacles = [
      [-0.5, 0.5, 0.5, 5],     # Obstacle 1
      [-0.5, 0.5, -5, -0.5],   # Obstacle 2
      [-5, 5, -5, -4.5],       # Obstacle 3
      [-5, 5, 4.5, 5],         # Obstacle 4
      [-5, -4.5, -5, 5],       # Obstacle 5
      [4.5, 5, -5, 5]          # Obstacle 6
    ]
    result = np.zeros((3,ic))
    #Change this when also changing g_y
    g_orig = np.array([-3,0,3])
    for k in range(g_orig.size):
      for i in range(ic):
        print(i)
        loop_continue = True  # Control flag for the outer loop
        state = self.sample_random_state(
            sample_inside_obs=False,
            sample_inside_tar=False,
            theta=None,
        )
        while loop_continue:
            g_y = g_orig[k]
            state = np.array(state)
            state[-1] = g_y
            # Checking if the target is reachable
            state_tensor = (torch.FloatTensor(state).to(self.device).unsqueeze(0))
            initial_q = q_func(state_tensor).item()
            if initial_q <= 0:
                break
            else:
                # Informed Sampling
                mean = g_y
                var = 0.1  # Starting variance
                lb = self.bounds[3,0]
                ub = self.bounds[3,1]
                while True:
                    sd = np.sqrt(var)
                    if sd>1e15:
                        print("No goal found")
                        result[k,i] = -1
                        loop_continue = False
                        break
                    a, b = (lb - mean) / sd, (ub - mean) / sd
                    truncated_gaussian = truncnorm(a, b, loc=mean, scale=sd)
                    samples = truncated_gaussian.rvs(100)
                    #print(samples)
                    #print(var)
                    found = False
                    for sample in samples:
                        state[-1] = sample
                        state_tensor = (torch.FloatTensor(state).to(self.device).unsqueeze(0))
                        initial_q = q_func(state_tensor).item()
                        #print(initial_q)
                        if initial_q <= 0:
                            g_y = sample
                            found = True
                            break
                    if found:
                        result[k,i] = abs(g_y-g_orig[k])
                        loop_continue=False
                        break
                    var *= 2
    return result