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
from robomimic.utils.file_utils import policy_from_checkpoint, env_from_checkpoint
from copy import deepcopy
from .env_utils import calculate_signed_distance
import time
import itertools
import imageio

class LiftPolicyEnv(gym.Env):
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
    env_id = cfg_env.env_id
    if env_id != "LiftDream":
      raise NotImplementedError("Only Reach environment is supported")
    keys = cfg_env.obs_keys
    self.keys = keys
    horizon = cfg_env.robosuite.horizon
    config_suite = OmegaConf.to_container(cfg_env.robosuite)
    env = GymWrapper(suite.make(env_id, **config_suite), keys=keys)
    self.gym_env = env
    self.observation_space = self.gym_env.observation_space

    checkpoint = f"{cfg_env.checkpoint_folder}/models/{cfg_env.checkpoint}"
    policy, ckpt_dict = policy_from_checkpoint(
            ckpt_path=checkpoint,
    )
    self.policy = policy
    
    self.render = render
    self.vis_callback = vis_callback
    self.render_offscreen = cfg_env.render_offscreen
    env, _ = env_from_checkpoint(ckpt_dict=ckpt_dict, render=self.render, render_offscreen=self.render_offscreen)    
    self.suite_env = env
    np.random.seed(None)
    self.state = self.suite_env.reset()
    self.timeout = self.suite_env.env.horizon
    
    #State Space to train VF on (20 dimensions)
    self.bounds = np.array([
        [-0.6, 0.6], #ee x pos (Table bounds (x))
        [-0.6, 0.6], #ee y pos (Table bounds (y))
        [0.79, 1.5], #ee z pos (Table bounds (z))
        [-1, 1], #quat x
        [-1, 1], #quat y
        [-1, 1], #quat z
        [-1, 1], #quat w
        [-0.4, 0.4], #gripper left state
        [-0.4, 0.4], #gripper right state
        [-0.4, 0.4], #obj x pos (Table bounds (x))
        [-0.4, 0.4], #obj y pos (Table bounds (y))
        [0.79, 0.88], #obj z pos (Table bounds (z))
        [-1, 1], #quat x
        [-1, 1], #quat y
        [-1, 1], #quat z
        [-1, 1], #quat w
        [-1, 1], #rel dist x
        [-1, 1], #rel dist y
        [-1, 1], #rel dist z
        [0, 2] #obj id
    ])
    self.low = self.bounds[:, 0]
    self.high = self.bounds[:, 1]
    self.midpoint = (self.low + self.high) / 2.0
    self.interval = self.high - self.low
    self.sample_inside_obs = sample_inside_obs
    self.sample_inside_tar = sample_inside_tar
    self.obs_dim = 20
    self.numActionList = [1,1,1,1,1,1,1]
    self.action_space = gym.spaces.Discrete(1) #Would be 1 for policies (1^7)

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
    if isinstance(state, dict):
      state = self.convert_obs_to_np(state)
    for dim, bound in enumerate(self.bounds):
      flagLow = state[dim] < bound[0]
      flagHigh = state[dim] > bound[1]
      # if flagLow or flagHigh:
      #   print("out of bounds")
      #   return False
      if flagLow or flagHigh:
          # if flagLow:
          #     print(f"Out of bounds: dimension {dim} below lower bound {bound[0]}")
          # if flagHigh:
          #     print(f"Out of bounds: dimension {dim} above upper bound {bound[1]}")
          return False
    return True
  
  def safety_margin(self, s):
      if isinstance(s, dict):
        s = self.convert_obs_to_np(s)
      #Enclosure Safety Margin
      g1 = np.sqrt(s[16]**2 + s[17]**2 + s[18]**2) - 0.12
      g2 = -1*(s[7])+0.003
      safety_margin = min(g1, g2)
      #g3 = 0.79 - s[2]
      #safety_margin = max(g3,safety_margin)
      return safety_margin

  def target_margin(self, s):
      if isinstance(s, dict):
        s = self.convert_obs_to_np(s)
      #Object height
      target_margin = -1*(s[11] - 0.87)
      return target_margin
  
  def _reset_suite(self, state=None):
        if state is not None:
            state = state.item()
            self.suite_env.reset_to(state)
        else:
            state = self.suite_env.reset()
        self.state = state
        return self.state
    
  def render_sim(self):
        self.suite_env.render(camera_name="agentview")

  def reset(self):
        obs = self._reset_suite()
        return obs
    
  def convert_obs_to_np(self, obs):
    # Initialize an empty NumPy array with the desired shape
    obs_np = np.zeros(self.observation_space.shape)
    total_shape = 0
    for key in self.keys:
        if key == "object-state":
            key = "object"
        obs_val = obs[key].flatten()
        shape = obs_val.shape[0]
        if isinstance(obs_val, torch.Tensor):
            obs_val = obs_val.detach().cpu().numpy()
        obs_np[total_shape:total_shape + shape] = obs_val
        total_shape += shape
    return obs_np
  
  def step(self, act=0):
    """Evolves the environment one step forward for manipulator.

    Returns:
        np.ndarray: next state.
        float: safety margin value for the next state.
        bool: True if the episode is terminated for any lander.
        dict: dictionary with safety and target margins.
    """
    # call policy
    action = self.policy(ob=self.state)
    # play action
    #Next State, Reward, Done (Task Success), Info
    next_obs, _, _, _ = self.suite_env.step(action)
    self.state = next_obs

    #Safety Margin (boundary check)
    l_x = self.target_margin(self.state)
    g_x = self.safety_margin(self.state)
    fail = g_x > 0
    success = l_x <= 0
    
    # cost
    if self.mode == "RA":
      if fail:
        cost = self.penalty
      elif success:
        cost = self.reward
      else:
        cost = 0.0
    else:
      if fail:
        cost = self.penalty
      elif success:
        cost = self.reward
      else:
        cost = 0

    # done
    if self.doneType == 'toEnd':
      done = not self.check_within_bounds(self.state)
    elif self.doneType == 'toFailureOrSuccess':
      if success or fail:
        done = True
    # = `info`
    if self.doneType == "fail":
      info = {"g_x": self.penalty * self.scaling}
    else:
      info = {"g_x": g_x, "l_x": l_x}
    return self.state, cost, done, info

  def simulate_one_trajectory(self, q_func, T=500, init_q=False, toEnd=False):
    start_time = time.time()
    self.policy.start_episode()
    # np.random.seed(None)
    # while True:
    #   obs = self.suite_env.reset()
    #   state_dict = self.suite_env.get_state()
    #   obs = self.suite_env.reset_to(state_dict)
    #   obsc = self.convert_obs_to_np(obs)
    #   state_tensor = torch.FloatTensor(obsc)
    #   state_tensor = state_tensor.to(self.device).unsqueeze(0)        
    #   initial_q = q_func(state_tensor).item()
    #   if(abs(initial_q) < 0.1):
    #     print(initial_q)
    #     break
    obs = self.suite_env.reset()
    state_dict = self.suite_env.get_state()
    obs = self.suite_env.reset_to(state_dict)
    result = 0
    states = []
    initial_q = None
    states.append(self.state)
    #print(self.convert_obs_to_np(obs))
    for _ in range(T):
        obs = self.convert_obs_to_np(obs)
        state_tensor = torch.FloatTensor(obs)
        state_tensor = state_tensor.to(self.device).unsqueeze(0)        
        if initial_q is None:
            initial_q = q_func(state_tensor).item()
        # step env
        next_obs, _, done, _ = self.step()
        obs = next_obs
        states.append(next_obs)
        # visualization
        if self.render:
            self.suite_env.render(mode="human", camera_name="agentview")
        l_x = self.target_margin(self.state)
        g_x = self.safety_margin(self.state)
        # break if done or if success
        if l_x <= 0:
            result = 1 #Success
            break
        if done or g_x > 0:
            result = -1 #Failed
        # update for next iter
    end_time = time.time()
    elapsed_time = end_time - start_time
    #print(f"Traj Execution time: {elapsed_time:.4f} seconds")
    # print("target margin ", l_x)
    # print("safety margin", g_x)
    # print("Initial q", initial_q)
    if result == 0:
      result = -1
    if init_q:
      return states, result, initial_q
    return states, result

  def simulate_trajectories(self, q_func, T=500, num_rnd_traj=None, toEnd=False):
    trajectories = []
    start_time = time.time()
    results = np.empty(shape=(num_rnd_traj,), dtype=int)
    for idx in range(num_rnd_traj):
        traj, result = self.simulate_one_trajectory(q_func=q_func, T=T)
        trajectories.append((traj))
        results[idx] = result
    end_time = time.time()
    elapsed_time = end_time - start_time
    #print(f"Traj Execution time: {elapsed_time:.4f} seconds")
    return trajectories, results
  
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
        states[i, :] = s
      else:
        l_x = self.target_margin(s)
        g_x = self.safety_margin(s)
        heuristic_v[i, :] = np.maximum(l_x, g_x)
        states[i, :] = s
    return states, heuristic_v

  def confusion_matrix(self, q_func, num_states=100):
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
      #print(ii)
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
  
  def ooa(self, q_func, video_path):
    self.policy.start_episode()
    np.random.seed(None)
    name_to_id = {'Red Bowl': 0, 'Brown Bowl': 1, 'Mug':2}
    obs = self.suite_env.reset()
    state_dict = self.suite_env.get_state()
    #Initial starting State
    obs = self.suite_env.reset_to(state_dict)
    saved_key = None
    # Call self.suite_env.reset_to until you get the right object id
    loop_continue = True
    while loop_continue:
      user_input = input("Enter object name (Red Bowl, Brown Bowl, or Mug): ")
      while True:
        if obs['object'][-1]==name_to_id[user_input]:
          break
        else:
          obs = self.suite_env.reset_to(state_dict)
      #Convert to VF friendly form
      obs = self.convert_obs_to_np(obs)
      state_tensor = torch.FloatTensor(obs)
      state_tensor = state_tensor.to(self.device).unsqueeze(0)
      # Checking if the target is reachable
      if q_func(state_tensor).item() <= 0:
          print("Target Reachable!")
          break
      else:
          # Try for every single object
          keys_not_user_input = [key for key in name_to_id if key != user_input]
          found = False
          for key in keys_not_user_input:
            #Reset until key is fully mapped
            while True:
              if obs[-1]==name_to_id[key]:
                break
              else:
                obs = self.suite_env.reset_to(state_dict)
                obs = self.convert_obs_to_np(obs)
            state_tensor = torch.FloatTensor(obs)
            state_tensor = state_tensor.to(self.device).unsqueeze(0)
            if q_func(state_tensor).item() <= 0:
              saved_key = key
              found = True
              break
          if found:
            ans = input(f"Current object isn't reachable, but {saved_key} is, accept (Y/N)? ")
            if ans == "Y":
              print("New object accepted.")
              loop_continue = False
              break
            elif ans == "N":
              print("Restarting, please propose a new object.")
              break
          else:
             print("None of the objects are reachable at this initial state.")
             exit()
    states = []
    video_count = 0
    video_skip = 5
    video_writer = imageio.get_writer(video_path, fps=20)
    result = 0
    for _ in range(self.timeout):
        state_tensor = torch.FloatTensor(obs)
        state_tensor = state_tensor.to(self.device).unsqueeze(0)        
        # step env
        next_obs, _, done, _ = self.step()
        obs = next_obs
        obs = self.convert_obs_to_np(obs)
        states.append(next_obs)
        # visualization
        if self.render:
          self.suite_env.render(mode="human", camera_name="agentview")
        if video_writer is not None:
          if video_count % video_skip == 0:
              video_img = []
              video_img.append(self.suite_env.render(mode="rgb_array", height=512, width=512, camera_name="agentview"))
              video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
              video_writer.append_data(video_img)
          video_count += 1
        l_x = self.target_margin(self.state)
        g_x = self.safety_margin(self.state)
        # break if done or if success
        if l_x <= 0:
            result = 1 #Success
            break
        if done or g_x > 0:
            result = -1 #Failed
        # update for next iter
    if result == 0:
      result = -1
    return states, result    
  

  def ooa_test(self, q_func, ic=1000):
    np.random.seed(None)
    name_to_id = {'Red Bowl': 0, 'Brown Bowl': 1, 'Mug':2}
    name_to_id = {'Brown Bowl': 1}
    result = np.zeros((3,ic))
    #Red bowl = 0, Brown Bowl = 1, and Mug = 2, Both (not proposed obj) = 3, None = -1
    for key_orig, k in name_to_id.items():
       for i in range(ic):
        print(i)
        self.policy.start_episode()
        obs = self.suite_env.reset()
        state_dict = self.suite_env.get_state()
        #Initial starting State
        obs = self.suite_env.reset_to(state_dict)
        while True:
          if obs['object'][-1]==name_to_id[key_orig]:
            break
          else:
            # Call self.suite_env.reset_to until you get the right object id
            obs = self.suite_env.reset_to(state_dict)
        loop_continue = True
        while loop_continue:
          #Convert to VF friendly form
          obs = self.convert_obs_to_np(obs)
          state_tensor = torch.FloatTensor(obs)
          state_tensor = state_tensor.to(self.device).unsqueeze(0)
          # Checking if the target is reachable
          if q_func(state_tensor).item() <= 0:
              result[k,i]=name_to_id[key_orig]
              print("Original works")
              break
          else:
              print("Finding Alternative")
              # Try for every single object
              keys_not_user_input = [key for key in name_to_id if key != key_orig]
              valid_keys = []
              for key in keys_not_user_input:
                #Reset until key is fully mapped
                while True:
                  if obs[-1]==name_to_id[key]:
                    break
                  else:
                    obs = self.suite_env.reset_to(state_dict)
                    obs = self.convert_obs_to_np(obs)
                state_tensor = torch.FloatTensor(obs)
                state_tensor = state_tensor.to(self.device).unsqueeze(0)
                if q_func(state_tensor).item() <= 0:
                  valid_keys.append(key)
              if len(valid_keys)==0:
                 result[k,i]=-1
              elif len(valid_keys)==1:
                 result[k,i]=name_to_id[valid_keys[0]]
              elif len(valid_keys)==2:
                 result[k,i]=3
              loop_continue = False
        print(result[k,i])          
    return result