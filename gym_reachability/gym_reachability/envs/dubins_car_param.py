import gym.spaces
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import torch
import random
from .env_utils import plot_arc, plot_circle
from .env_utils import calculate_margin_circle, calculate_margin_rect

class DubinsCarParamEnv(gym.Env):
  """A gym environment considering Dubins car dynamics.
  """

  def __init__(
      self, device, mode="RA", doneType="toEnd", sample_inside_obs=True,
      sample_inside_tar=True
  ):
    """Initializes the environment with given arguments.

    Args:d
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
    self.bounds = np.array([[-5, 5], [-5, 5], [0, 2 * np.pi]]) #x, y, th, param
    self.low = self.bounds[:, 0]
    self.high = self.bounds[:, 1]
    self.sample_inside_obs = sample_inside_obs
    self.sample_inside_tar = sample_inside_tar
    self.constraint_center = None
    self.constraint_radius = None

    # Gym variables.
    self.action_space = gym.spaces.Discrete(3) #{-2, 2}
    self.midpoint = (self.low + self.high) / 2.0
    self.interval = self.high - self.low
    self.observation_space = gym.spaces.Box(
        np.float32(self.midpoint - self.interval/2),
        np.float32(self.midpoint + self.interval/2),
    )

    # Constraint set parameters.
    self.constraint_x_y_w_h = np.array([
          [0., -4.75, 10., 0.5], #Top Obstacle
          [0., 4.75, 10., 0.5], #Bottom Obstacle
          [-4.75, 0, 0.5, 10.], #Left Obstacle
          [4.75, 0., 0.5, 10.], #Right Obstacle
          [0, 3, 2.75, 5], #Top Narrow
          [0, -3, 2.75, 5], #Bottom Narrow  
    ])

    # Target set parameters.
    self.target_center = np.array([3, 0])
    self.target_radius = 0.5

    # Internal state.
    self.mode = mode
    self.state = np.zeros(3)
    self.doneType = doneType

    # Dubins car parameters.
    self.time_step = 0.1
    self.speed = 1
    self.discrete_controls = np.array([
        -2, 0, 2
    ])

    # Visualization params
    self.visual_initial_states = [
        np.array([-3, 3, 0]),
        np.array([-3, 0, np.pi / 2]),
        np.array([-3, -3, np.pi])
    ]

    # Cost Params
    self.targetScaling = 1.
    self.safetyScaling = 1.
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
      x_rnd, y_rnd, theta_rnd = self.sample_random_state(
          sample_inside_obs=sample_inside_obs,
          sample_inside_tar=sample_inside_tar, theta=theta
      )
      self.state = np.array([x_rnd, y_rnd, theta_rnd])
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
        low=self.low, high=self.high, size=(num_warmup_samples, 3)
    )
    x_rnd, y_rnd, theta_rnd = rv[:, 0], rv[:, 1], rv[:, 2]

    heuristic_v = np.zeros((num_warmup_samples, self.action_space.n))
    states = np.zeros((num_warmup_samples, self.observation_space.shape[0]))

    for i in range(num_warmup_samples):
      x, y, theta = x_rnd[i], y_rnd[i], theta_rnd[i]
      l_x = self.target_margin(np.array([x, y]))
      g_x = self.safety_margin(np.array([x, y]))
      heuristic_v[i, :] = np.maximum(l_x, g_x)
      states[i, :] = x, y, theta

    return states, heuristic_v

  def get_value(self, q_func, theta, nx=101, ny=101, addBias=False):
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
      l_x = self.target_margin(np.array([x, y]))
      g_x = self.safety_margin(np.array([x, y]))

      if self.mode == "normal" or self.mode == "RA":
        state = (torch.FloatTensor([x, y, theta]).to(self.device).unsqueeze(0))
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
    # random sample `theta`
    if theta is None:
      theta_rnd = 2.0 * np.random.uniform() * np.pi
    else:
      theta_rnd = theta

    # random sample [`x`, `y`]
    flag = True
    while flag:
      rnd_state = np.random.uniform(low=self.low[:2], high=self.high[:2])
      l_x = self.target_margin(rnd_state)
      g_x = self.safety_margin(rnd_state)

      if (not sample_inside_obs) and (g_x > 0):
        flag = True
      elif (not sample_inside_tar) and (l_x <= 0):
        flag = True
      else:
        flag = False
    x_rnd, y_rnd = rnd_state

    return x_rnd, y_rnd, theta_rnd

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
    l_x = self.target_margin(self.state[:2])
    g_x = self.safety_margin(self.state[:2])

    fail = g_x > 0
    success = l_x <= 0

    u = self.discrete_controls[action]
    state = self.integrate_forward(self.state, u)
    self.state = state

    # cost
    if self.mode == "RA":
      if fail:
        cost = self.penalty
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
      info = {"g_x": g_x, "l_x": l_x}

    return np.copy(self.state), cost, done, info

  def integrate_forward(self, state, u):
    """Integrates the dynamics forward by one step.

    Args:
        state (np.ndarray): (x, y, yaw).
        u (float): the contol input, angular speed.

    Returns:
        np.ndarray: next state.
    """
    x, y, theta = state

    x = x + self.time_step * self.speed * np.cos(theta)
    y = y + self.time_step * self.speed * np.sin(theta)
    theta = np.mod(theta + self.time_step * u, 2 * np.pi)

    state = np.array([x, y, theta])
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

  def set_speed(self, speed=0.5):
    """Sets the linear velocity of the car.

    Args:
        speed (float, optional): speed of the car. Defaults to .5.
    """
    self.speed = speed
  
  def set_target(self, center, radius):
    """Sets the target set.

    Args:
        center (np.ndarray, optional): center of the target set.
        radius (float, optional): radius of the target set.
    """
    self.target_center = center
    self.target_radius = radius

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
    if self.target_center is not None and self.target_radius is not None:
      target_margin = calculate_margin_circle(
          s, [self.target_center, self.target_radius], negativeInside=True
      )
      return self.targetScaling * target_margin
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
      self, q_func, T=250, state=None, theta=None, sample_inside_obs=True,
      sample_inside_tar=True, toEnd=False
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
    traj = []
    result = 0  # not finished
    valueList = []
    gxList = []
    lxList = []
    for t in range(T):
      traj.append(state)

      g_x = self.safety_margin(state)
      l_x = self.target_margin(state)

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
        done = not self.check_within_bounds(state)
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

      q_func.eval()
      state_tensor = (torch.FloatTensor(state).to(self.device).unsqueeze(0))
      action_index = q_func(state_tensor).min(dim=1)[1].item()
      u = self.discrete_controls[action_index]

      state = self.integrate_forward(state, u)
    traj = np.array(traj)
    info = {"valueList": valueList, "gxList": gxList, "lxList": lxList}
    return traj, result, minV, info

    return trajectories, results, minVs

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
      nx = 41
      ny = nx
      xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
      ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
      results = np.empty((nx, ny), dtype=int)
      minVs = np.empty((nx, ny), dtype=float)

      it = np.nditer(results, flags=["multi_index"])
      print()
      while not it.finished:
        idx = it.multi_index
        print(idx, end="\r")
        x = xs[idx[0]]
        y = ys[idx[1]]
        state = np.array([x, y, 0.0])
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
      labels=None, boolPlot=False, addBias=False, theta=np.pi / 2,
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
    thetaList = [np.pi / 2, 0, -np.pi / 2]
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    axList = [ax1, ax2, ax3]

    for i, (ax, theta) in enumerate(zip(axList, thetaList)):
      # for i, (ax, theta) in enumerate(zip(self.axes, thetaList)):
      ax.cla()
      if i == len(thetaList) - 1:
        cbarPlot = True
      else:
        cbarPlot = False

      # == Plot failure / target set ==
      self.plot_target_failure_set(ax)

      # == Plot reach-avoid set ==
      #self.plot_reach_avoid_set(ax, orientation=theta)

      # == Plot V ==
      self.plot_v_values(
          q_func,
          ax=ax,
          fig=fig,
          theta=theta,
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
            toEnd=False,
            ax=ax,
            c="y",
            lw=2,
            orientation=theta - np.pi / 2,
        )

      ax.set_xlabel(
          r"$\theta={:.0f}^\circ$".format(theta * 180 / np.pi),
          fontsize=28,
      )

    plt.tight_layout()

  def plot_v_values(
      self, q_func, theta=np.pi / 2, ax=None, fig=None, vmin=-1, vmax=1,
      nx=201, ny=201, cmap="seismic", boolPlot=False, cbarPlot=True,
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
      theta = 2.0 * np.random.uniform() * np.pi
    v = self.get_value(q_func, theta, nx, ny, addBias=addBias)

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

  def plot_trajectories(
      self, q_func, T=250, num_rnd_traj=None, states=None, theta=None,
      toEnd=False, ax=None, c="y", lw=1.5, orientation=np.pi/2, zorder=2
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
        tmpStates.append(np.array([xtilde, ytilde, thetatilde]))
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

  def plot_target_failure_set(self, ax=None, c_c="m", c_t="y", lw=3, zorder=0):
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
        self.target_center,
        self.target_radius,
        ax,
        c=c_t,
        lw=lw,
        zorder=zorder,
    )
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
