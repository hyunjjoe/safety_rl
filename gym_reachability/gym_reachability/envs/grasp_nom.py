import gym.spaces
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import torch
import random
import robosuite as suite
from robosuite.wrappers import GymWrapper
from omegaconf import OmegaConf
from robosuite.environments.manipulation.grasp import Grasp
from copy import deepcopy
from .env_utils import calculate_margin_cube, calculate_signed_distance

class GraspNomEnv(gym.Env):
  """Wrapper class for the Grasp env in Robosuite
  """

  def __init__(
      self, device, cfg_env, mode="RA", doneType="toEnd", sample_inside_obs=True,
      sample_inside_tar=True, render=False, vis_callback=None
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
    self.seed_val = cfg_env.seed
    env_id = cfg_env.env_id
    if env_id != "Grasp":
      raise NotImplementedError("Only Grasp environment is supported")
    keys = cfg_env.obs_keys
    self.keys = keys
    self.horizon = cfg_env.robosuite.horizon
    config_suite = OmegaConf.to_container(cfg_env.robosuite)
    env = GymWrapper(suite.make(env_id, **config_suite), keys=keys)
    self.gym_env = env
    self.observation_space = self.gym_env.observation_space
    checkpoint = f"{cfg_env.checkpoint_folder}/models/{cfg_env.checkpoint}"

    self.render = render
    self.vis_callback = vis_callback  
    self.suite_env = env
    self.state, _ = self.suite_env.reset()
    self.timeout = self.suite_env.env.horizon
    
    #State Space to train VF on (16 dimensions)
    self.bounds = np.array([
        [-0.4, 0.4], #ee x pos (Table bounds (x))
        [-0.4, 0.4], #ee y pos (Table bounds (y))
        [0.81, 1.2], #ee z pos (Table bounds (z))
        [-1, 1], #quat x
        [-1, 1], #quat y
        [-1, 1], #quat z
        [-1, 1], #quat w
        [-0.4, 0.4], #gripper x pos
        [-0.4, 0.4], #gripper y pos
        [-0.4, 0.4], #cube x pos (Table bounds (x))
        [-0.4, 0.4], #cube y pos (Table bounds (y))
        [0.81, 0.84], #cube z pos (Table bounds (z))
        [-1, 1], #quat x
        [-1, 1], #quat y
        [-1, 1], #quat z
        [-1, 1], #quat w
    ])
    self.low = self.bounds[:, 0]
    self.high = self.bounds[:, 1]
    self.midpoint = (self.low + self.high) / 2.0
    self.interval = self.high - self.low
    self.obs_dim = 16
    self.object_dims = (0.02, 0.02, 0.02)
    self.numActionList = [3,3,3,3,3,3,3]
    self.action_space = gym.spaces.Discrete(3**7)
    self.discrete_controls = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
        ])
    self.sample_inside_obs = sample_inside_obs
    self.sample_inside_tar = sample_inside_tar
        
    #Internal state
    self.mode = mode
    self.doneType = doneType

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
  
  def safety_margin(self, s):
      #s = self.convert_obs_to_np(s)
      #Enclosure Safety Margin
      g_x_list = []
      boundary = np.append(self.midpoint[:3], self.interval[:3])
      g_x = calculate_margin_cube(s, boundary, negativeInside=True)
      g_x_list.append(g_x)
      safety_margin = np.max(np.array(g_x_list))      
      return safety_margin

  def target_margin(self, s):
      #s = self.convert_obs_to_np(s)
      #Solely Cube and EE rel pos
      target_margin = calculate_signed_distance(s[:3], s[10:13], self.object_dims)
      return target_margin

  def _reset_suite(self, state=None):
        if state is not None:
            state = state.item()
            self.suite_env.reset_to(state)
        else:
            state, _ = self.suite_env.reset()
        self.state = state
        return self.state
    
  def render_sim(self):
        self.suite_env.render(camera_name="frontview")

  def reset(self):
        obs = self._reset_suite()
        # return self.convert_obs_to_np(obs)
        return obs
    
  def convert_obs_to_np(self, obs):
        # loop through keys in proper order and create numpy observation
        obs_np = np.zeros(self.observation_space.shape)
        total_shape = 0
        for key in sorted(self.keys):
            if key == "object-state":
                key = "object"
            shape = obs[key].flatten().shape[0]
            obs_val = obs[key].flatten()
            if isinstance(obs_val, torch.Tensor):
                obs_val = obs_val.detach().cpu().numpy()
            obs_np[total_shape:total_shape+shape] = obs_val
            total_shape += shape
        return obs_np
  
  def step(self, act):
    """Evolves the environment one step forward for manipulator.

    Returns:
        np.ndarray: next state.
        float: safety margin value for the next state.
        bool: True if the episode is terminated for any lander.
        dict: dictionary with safety and target margins.
    """
    #Target Margin
    l_x = self.target_margin(self.state)
    #Safety Margin (boundary check)
    g_x = self.safety_margin(self.state)
    fail = g_x > 0
    success = l_x <= 0

    # play action
    # Decode the action index into discrete controls
    act_indices = np.unravel_index(act, self.numActionList)
    act = self.discrete_controls[np.arange(7), act_indices]
    next_obs, reward, done, _, info = self.suite_env.step(act)
    self.state = next_obs
    
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

    # = `info`
    if done and self.doneType == "fail":
      info = {"g_x": self.penalty * self.scaling}

    return self.state, cost, done, info

  def simulate_one_trajectory(self, q_func, init_q=False):
    obs = self.suite_env.reset()
    state_dict = self.suite_env.get_state()
    obs = self.suite_env.reset_to(state_dict)
    result = 0
    traj = dict(actions=[], dones=[], states=[], initial_state_dict=state_dict)
    initial_q = None

    for _ in range(self.timeout):
        # get obs state in np
        state_np = self.convert_obs_to_np(obs)
        state_tensor = torch.FloatTensor(state_np)
        state_tensor = state_tensor.to(self.device).unsqueeze(0)
        with torch.no_grad():
            state_action_values = q_func(state_tensor)
        if initial_q is None:
            initial_q = q_func(state_tensor).min(dim=1)[0].item()
        #action 
        Q_mtx = state_action_values.view(3, 3, 3, 3, 3, 3, 3)
        # Reduce dimensions to find the argmax values
        # Iterate through the dimensions to find the maximum values and indices
        min_values, min_indices = Q_mtx.min(dim=-1)
        for dim in range(len(self.numActionList) - 2, -1, -1):
            min_values, min_indices = min_values.min(dim=dim)
            min_indices = min_indices.view(-1)
        act = min_indices
        # play action
        next_obs, _, done, _ = self.step(act)
        l_x = self.target_margin(self.state)
        g_x = self.safety_margin(self.state)
        # visualization
        if self.render:
            self.suite_env.render(mode="human", camera_name="frontview")
        # collect transition
        traj["actions"].append(act)
        traj["dones"].append(done)
        traj["states"].append(state_dict["states"])
        # break if done or if success
        if done or l_x <= 0:
            result = 1 #Success
            break
        if g_x > 0:
            result = -1 #Failed
        # update for next iter
        obs = deepcopy(next_obs)
        state_dict = self.suite_env.get_state()
    if result == 0:
       result = -1
    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])
    if init_q:
      return traj, result, initial_q
    return traj, result

  def simulate_trajectories(self, q_func, num_traj):
    trajectories = []
    results = np.empty(shape=(num_traj,), dtype=int)
    for idx in range(num_traj):
        traj, result = self.simulate_one_trajectory(q_func)
        trajectories.append((traj))
        results[idx] = result
    return trajectories
  
  def get_warmup_examples(self, num_warmup_samples=100, s_margin=False):
    """Gets warmup samples to initialize the Q-network.

    Args:
        num_warmup_samples (int, optional): # warmup samples. Defaults to 100.
        s_margin (bool, optional): use safety margin as heuristic values if
            True. If False, use max{ell, g} instead. Defaults to true.

    Returns:
        np.ndarray: sampled states.
        np.ndarray: the heuristic values.
    """
    rv = np.random.uniform(
        low=self.bounds[:, 0], high=self.bounds[:, 1],
        size=(num_warmup_samples, self.obs_dim)
    )

    heuristic_v = np.zeros((num_warmup_samples, self.action_space.n))
    states = np.zeros((num_warmup_samples, self.observation_space.shape[0]))

    for i in range(num_warmup_samples):
      s = np.array(rv[i, :])
      if s_margin:
        g_x = self.safety_margin(s)
        heuristic_v[i, :] = g_x
        states[i, :] = self.convert_obs_to_np(s)
      else:
        l_x = self.target_margin(s)
        g_x = self.safety_margin(s)
        heuristic_v[i, :] = np.maximum(l_x, g_x)
        states[i, :] = self.convert_obs_to_np(s)
    return states, heuristic_v

  def confusion_matrix(self, q_func, num_states=50):
    """Gets the confusion matrix using DDQN values and rollout results.

    Args:
        q_func (object): agent's Q-network.
        num_states (int, optional): # initial states to rollout a trajectoy.
            Defaults to 50.

    Returns:
        np.ndarray: confusion matrix.
    """
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