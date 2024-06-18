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

class GraspEnv(gym.Env):
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
    self.suite_env = env

    checkpoint = f"{cfg_env.checkpoint_folder}/models/{cfg_env.checkpoint}"
    model, ckpt_dict = policy_from_checkpoint(
            ckpt_path=checkpoint,
    )
    
    self.render = render
    self.vis_callback = vis_callback
    env, _ = env_from_checkpoint(ckpt_dict=ckpt_dict, render=self.render)    


    print(
        "Env: mode-{:s}; doneType-{:s}; sample_inside_obs-{}".format(
            self.mode, self.doneType, self.sample_inside_obs
        )
    )

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
    
