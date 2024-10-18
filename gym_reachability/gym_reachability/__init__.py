"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
         Vicenc Rubies Royo ( vrubies@berkeley.edu )
"""

from gym.envs.registration import register

register(
    id="dubins_policy-v0",
    entry_point="gym_reachability.gym_reachability.envs:DubinsPolicyEnv"
)

register(
    id="lift_policy-v0",
    entry_point="gym_reachability.gym_reachability.envs:LiftPolicyEnv"
)