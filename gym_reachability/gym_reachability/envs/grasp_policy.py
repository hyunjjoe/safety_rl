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
from robomimic.utils.file_utils import policy_from_checkpoint, env_from_checkpoint
from copy import deepcopy
from .env_utils import calculate_margin_cube

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
    horizon = cfg_env.robosuite.horizon
    config_suite = OmegaConf.to_container(cfg_env.robosuite)
    env = GymWrapper(suite.make(env_id, **config_suite), keys=keys)
    self.gym_env = env
    self.observation_space = self.gym_env.observation_space
    self.action_space = self.gym_env.action_space

    checkpoint = f"{cfg_env.checkpoint_folder}/models/{cfg_env.checkpoint}"
    policy, ckpt_dict = policy_from_checkpoint(
            ckpt_path=checkpoint,
    )
    self.policy = policy
    
    self.render = render
    self.vis_callback = vis_callback
    env, _ = env_from_checkpoint(ckpt_dict=ckpt_dict, render=self.render)    
    self.suite_env = env
    self.state = self.suite_env.reset()
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

    #Failure set: Boundary of the state space

    print(
        "Env: mode-{:s}; doneType-{:s}; sample_inside_obs-{}".format(
            self.mode, self.doneType, self.sample_inside_obs
        )
    )
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
  
  def safety_margin(self, s):
      #Enclosure Safety Margin
      g_x_list = []
      boundary = np.append(self.midpoint[:3], self.interval[:3])
      g_x = calculate_margin_cube(s, boundary, negativeInside=True)
      g_x_list.append(g_x)
      safety_margin = np.max(np.array(g_x_list))      
      return safety_margin

  def _reset_suite(self, state=None):
        if state is not None:
            state = state.item()
            self.suite_env.reset_to(state)
        else:
            state = self.suite_env.reset()
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
    #Target Margin (call the success check)
    success = self.suite_env.is_success()["task"]
    #Safety Margin (boundary check)
    g_x = self.safety_margin(self.state)
    fail = g_x > 0

    # play action
    next_obs, _, done, _ = self.suite_env.step(act)
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

  def simulate_one_trajectory(self, q_func):
    self.policy.start_episode()
    obs = self.suite_env.reset()
    state_dict = self.suite_env.get_state()
    obs = self.suite_env.reset_to(state_dict)

    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)

    for _ in range(self.timeout):
        # get action from policy
        act = self.policy(ob=obs)
        # play action
        next_obs, _, done, _ = self.step(act)
        success = self.suite_env.is_success()["task"]
        # visualization
        if self.render:
            self.suite_env.render(mode="human", camera_name="frontview")
        # collect transition
        traj["actions"].append(act)
        traj["dones"].append(done)
        traj["states"].append(state_dict["states"])
        # break if done or if success
        if done or success:
            break
        # update for next iter
        obs = deepcopy(next_obs)
        state_dict = self.suite_env.get_state()
    
    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])
    return traj


  def simulate_trajectories(self, q_func, num_traj):
    trajectories = []
    results = np.empty(shape=(num_traj,), dtype=int)
    for idx in range(num_traj):
        traj_x, traj_y, result = self.simulate_one_trajectory(q_func)
        trajectories.append((traj_x, traj_y))
        results[idx] = result
    return trajectories