import numpy as np
import json
import tensorflow as tf
import os
import sys
import time
import gym

import tensorflow as tf

'''gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], 
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=256)])
  except RuntimeError as e:
    print(e)
    
    
print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.list_logical_devices('GPU')

'''

import tf_agents
from tf_agents.environments import py_environment, parallel_py_environment, batched_py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
#from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils.common import function, element_wise_squared_loss
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.policies import policy_saver
import logging

import tensorflow.keras as keras

tf.compat.v1.enable_v2_behavior()
import datetime
import copy
import csv

from scan_gym import envs
#seed=42
#tf.random.set_seed(seed)
#np.random.seed(seed)


def run_episode(env,tf_env,policy):
    state = tf_env.reset()
    time_steps = 1000
    actions = []
    images = []
    rewards = []
    images.append([env.current_theta,env.current_phi])
    for i in range(1,time_steps):
        action_step = policy.action(state)
        actions.append(int(action_step.action.numpy()[0]))
        state = tf_env.step(action_step.action)
        
        images.append([env.current_theta,env.current_phi])
        rewards.append(float(state.reward.numpy()[0]))
        if state.is_last():
            break
    return actions, images, env.theta_bias, rewards, env.total_reward , env.spc.gt_compare_solid(),env.num_steps
    #return [], images, env.theta_bias, [], env.total_reward , env.spc.gt_compare_solid()


def test_policy(environment, models_path, models, policy, n_images, n_episodes, dest_path ):
    data_template = {'actions':[],'images':[],'theta_biases':[],'rewards':[],'cum_rewards':[], 'gt_ratios':[], 'num_steps':[], 'cum_reward_mean':0.0, 'cum_reward_std':0.0, 'gt_ratio_mean':0.0,'gt_ratio_std':0}
    collected_data = {}

    for p in models:
        #load environment with selected plant model
        scan_env = suite_gym.load(environment, gym_kwargs={'models_path':models_path, 'train_models':[p],
                                                   'n_images':n_images, 'continuous':False,
                                                           'gt_mode':True,'cube_view':'static','multi_input':False})
        
        tf_env = tf_py_environment.TFPyEnvironment( scan_env )
        
        #run 180 times 
        data_holder = copy.deepcopy(data_template)
        
        for i in range(n_episodes):
            actions, images, theta_bias, rewards, t_rwd, gt_ratio, num_steps = run_episode(scan_env,tf_env,policy)
            data_holder['actions'].append(actions)
            data_holder['images'].append(images)
            data_holder['theta_biases'].append(theta_bias)
            data_holder['rewards'].append(rewards)
            data_holder['cum_rewards'].append(t_rwd)
            data_holder['gt_ratios'].append(gt_ratio)
            data_holder['num_steps'].append(num_steps)
            
            print("\rplant-{} {}     ".format(p, i), end="")

        data_holder['cum_reward_mean'] = float(np.mean(data_holder['cum_rewards']))
        data_holder['cum_reward_std'] = float(np.std(data_holder['cum_rewards']))
        
        data_holder['gt_ratio_mean'] = float(np.mean(data_holder['gt_ratios']))
        data_holder['gt_ratio_std'] = float(np.std(data_holder['gt_ratios']))

        collected_data[p] = copy.deepcopy(data_holder)

        print("{} rwd_mean {:.4f} rwd_std {:.4f} gt_ratio_mean {:.4f} gt_ratio_std {:.4f} s{}".format(p,data_holder['cum_reward_mean'],data_holder['cum_reward_std'], \
                                                                                                      data_holder['gt_ratio_mean'], data_holder['gt_ratio_std'],np.mean(data_holder['num_steps'])) )
    print('----')
    if dest_path != "":
        f = open(dest_path,'w')
        json.dump(collected_data, f)
        f.close()
        
