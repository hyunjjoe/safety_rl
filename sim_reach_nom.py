from warnings import simplefilter
import time
import os.path
import pickle
import argparse

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch
from omegaconf import OmegaConf

from RARL.DDQNSingle import DDQNSingle
from RARL.config import dqnConfig
from gym_reachability import gym_reachability  # Custom Gym env.
from omegaconf import OmegaConf

timestr = time.strftime("%Y-%m-%d-%H_%M_%S")
simplefilter(action='ignore', category=FutureWarning)

# == ARGS ==
parser = argparse.ArgumentParser()

parser.add_argument(
    "-nt", "--num_test", help="the number of tests", default=1, type=int
)
parser.add_argument(
    "-nw", "--num_worker", help="the number of workers", default=1, type=int
)
parser.add_argument(
    "-test", "--test", help="test a neural network", action="store_true"
)
parser.add_argument(
    "-rnd", "--randomSeed", help="random seed", default=0, type=int
)
parser.add_argument(
    "-dt", "--doneType", help="when to raise done flag", default='toEnd',
    type=str
)
parser.add_argument(
    "-p", "--path", help="path to model", default='toEnd', type=str
)
parser.add_argument(
    "-cfg", "--config_path", help="path to CONFIG.pkl file",
    default='toEnd', type=str
)
parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str,
      default=os.path.join("config", "reach.yaml")
)


# training scheme
parser.add_argument(
    "-w", "--warmup", help="warmup Q-network", action="store_true"
)
parser.add_argument(
    "-wi", "--warmupIter", help="warmup iteration", default=10000, type=int
)
parser.add_argument(
    "-ab", "--addBias", help="add bias term for RA", action="store_true"
)
parser.add_argument(
    "-mu", "--maxUpdates", help="maximal #gradient updates", default=5e6,
    type=int
)
parser.add_argument(
    "-ms", "--maxSteps", help="maximal length of rollouts", default=200,
    type=int
)
parser.add_argument(
    "-mc", "--memoryCapacity", help="memoryCapacity", default=150000, type=int
)
parser.add_argument(
    "-cp", "--checkPeriod", help="check the success ratio", default=50000,
    type=int
)
parser.add_argument(
    "-upe", "--update_period_eps", help="update period for eps scheduler",
    default=500000, type=int
)
parser.add_argument(
    "-upg", "--update_period_gamma", help="update period for gamma scheduler",
    default=500000, type=int
)
parser.add_argument(
    "-upl", "--update_period_lr", help="update period for lr cheduler",
    default=250000, type=int
)

# hyper-parameters
parser.add_argument(
    "-lr", "--learningRate", help="learning rate", default=1e-4, type=float
)
parser.add_argument(
    "-g", "--gamma", help="contraction coeff.", default=0.9, type=float
)
parser.add_argument(
    "-e", "--eps", help="exploration coeff.", default=0.5, type=float
)
parser.add_argument(
    "-arc", "--architecture", help="neural network architecture",
    default=[1024, 1024, 1024], nargs="*", type=int
)
#Tanh?
parser.add_argument(
    "-act", "--actType", help="activation type", default='ReLU', type=str
)
parser.add_argument("-dbl", "--double", help="double DQN", action="store_true")
parser.add_argument(
    "-bs", "--batchsize", help="batch size", default=100, type=int
)

# RL type
parser.add_argument("-m", "--mode", help="mode", default='RA', type=str)

# file
parser.add_argument(
    "-st", "--showTime", help="show timestr", action="store_true"
)
parser.add_argument("-n", "--name", help="extra name", default='', type=str)
parser.add_argument(
    "-of", "--outFolder", help="output file",
    default='experiments/Reach_nom' + timestr, type=str
)
parser.add_argument(
    "-pf", "--plotFigure", help="plot figures", action="store_true"
)
parser.add_argument(
    "-sf", "--storeFigure", help="store figures", action="store_true"
)


args = parser.parse_args()

# == CONFIGURATION ==
env_name = "reach_nom-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fn = args.name + '-' + args.doneType
if args.showTime:
  fn = fn + '-' + timestr

outFolder = os.path.join(args.outFolder, 'ReachNom-DDQN', fn)
print(outFolder)

CONFIG = dqnConfig(
    ENV_NAME=env_name,
    DEVICE=device,
    SEED=args.randomSeed,
    MAX_UPDATES=args.maxUpdates,  # Number of grad updates.
    MAX_EP_STEPS=args.maxSteps,  # Max number of steps per episode.
    # =================== EXPLORATION PARAMS.
    EPSILON=args.eps,  # Initial exploration rate.
    EPS_END=0.05,  # Final explortation rate.
    EPS_PERIOD=args.update_period_eps / 10,  # How often to update EPS.
    EPS_DECAY=0.8,  # Rate of decay.
    EPS_RESET_PERIOD=args.update_period_eps,
    # =================== LEARNING RATE PARAMS.
    LR_C=args.learningRate,  # Learning rate.
    LR_C_END=args.learningRate * 0.5,  # Final learning rate.
    LR_C_PERIOD=args.update_period_lr,  # How often to update lr.
    LR_C_DECAY=0.8,  # Learning rate decay rate.
    # =================== LEARNING RATE .
    GAMMA=args.gamma,  # Inital gamma.
    GAMMA_END=0.99999,  # Final gamma.
    GAMMA_PERIOD=args.update_period_gamma,  # How often to update gamma.
    GAMMA_DECAY=0.1,  # Rate of decay of gamma.
    # ===================
    TAU=0.01,
    HARD_UPDATE=1,
    SOFT_UPDATE=True,
    MEMORY_CAPACITY=args.memoryCapacity,
    BATCH_SIZE=args.batchsize,  # Number of examples to use to update Q.
    RENDER=False,
    MAX_MODEL=10,  # How many models to store while training.
    DOUBLE=args.double,
    ARCHITECTURE=args.architecture,
    ACTIVATION=args.actType,
    REWARD=-1,
    PENALTY=0.1
)

# == REPORT ==
def report_config(CONFIG):
  for key, value in CONFIG.__dict__.items():
    if key[:1] != '_':
      print(key, value)

# == LOAD CONFIG FILE ==
cfg = OmegaConf.load(args.config_file)

# == ENVIRONMENT ==
env = gym.make(
    env_name, device=device, cfg_env=cfg.environment, mode="RA", doneType='toEnd'
)

# == EXPERIMENT ==
def run_experiment(args, CONFIG, env):
  """Run the reach-avoid training algorithm.

  Args:
      args: parsed arguments.
      CONFIG (config, object): configuration parameters.
      env (gym.Env): environment used for training.
  """
  # == AGENT ==
  s_dim = env.observation_space.shape[0]
  numAction = env.action_space.n
  actionList = np.arange(numAction)
  dimList = [s_dim] + args.architecture + [numAction]

  np.random.seed(args.randomSeed)
  agent = DDQNSingle(CONFIG, numAction, actionList, dimList, mode='RA')
  _, trainProgress = agent.learn(
      env,
      MAX_UPDATES=CONFIG.MAX_UPDATES,
      MAX_EP_STEPS=CONFIG.MAX_EP_STEPS,
      warmupBuffer=False,
      warmupQ=args.warmup,
      warmupIter=args.warmupIter,
      addBias=args.addBias,
      doneTerminate=True,
      runningCostThr=None,
      curUpdates=None,
      plotFigure=args.plotFigure,  # Display value function while learning.
      showBool=False,  # Show boolean reach avoid set 0/1.
      vmin=-1,
      vmax=1,
      checkPeriod=args.checkPeriod,  # How often to compute Safe vs. Unsafe.
      storeFigure=args.storeFigure,  # Store the figure in an eps file.
      storeModel=True,
      storeBest=False,
      outFolder=outFolder,
      verbose=True
  )
  return trainProgress

run_experiment(args, CONFIG, env)

