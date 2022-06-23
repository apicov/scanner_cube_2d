#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/pico/anaconda3/envs/rl/lib')


# In[2]:


import base64
import imageio
import IPython
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import os
import reverb
import tempfile
import PIL.Image
import numpy as np

import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], 
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])
  except RuntimeError as e:
    print(e)
    
    
print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.list_logical_devices('GPU')



from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
#from tf_agents.environments import suite_pybullet
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

from tf_agents.metrics import tf_metrics
import tensorflow.keras as keras

tf.compat.v1.enable_v2_behavior()
import time
import json
import datetime
import copy
import shutil

#import imp
from scan_gym import envs
#imp.reload(envs)

seed=43
tf.random.set_seed(seed)
np.random.seed(seed)

from utils import policy_test_sac

tempdir = tempfile.gettempdir()
import gym
from tf_agents.environments import suite_gym

from tf_agents.environments import random_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.specs import array_spec
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils
tf.compat.v1.enable_v2_behavior()


# In[2]:


#! PATH=$PATH:/home/pico/anaconda3/envs/rl/lib
#get_ipython().system('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/pico/anaconda3/envs/rl/lib')
#get_ipython().system('echo $LD_LIBRARY_PATH')


# In[3]:


#get_ipython().system('echo $LD_LIBRARY_PATH')


# In[2]:


class ActorNetworkCustom(network.Network):
    def __init__(self,
                observation_spec,
                action_spec,
                name='ActorNetworkCustom'):
        
        super(ActorNetworkCustom, self).__init__(
            input_tensor_spec=observation_spec, state_spec=(), name=name)

        # For simplicity we will only support a single action float output.
        self._action_spec = action_spec
        flat_action_spec = tf.nest.flatten(action_spec)
        self._single_action_spec = flat_action_spec[0]

        # Initialize the custom tf layers here:
        self._preprocessing = tf.keras.layers.Lambda(lambda x: (tf.cast(x,np.float32) / 255))
        self._conv1 = keras.layers.Conv2D(filters=16, kernel_size=3,strides=2, padding="same", activation="relu")
        self._conv2 = keras.layers.Conv2D(filters=32, kernel_size=3,strides=2, padding="same", activation="relu")
        self._conv3 = keras.layers.Conv2D(filters=64, kernel_size=3,strides=2,padding="same", activation="relu")
        #self._flatten = keras.layers.Flatten()
        self._flatten = keras.layers.GlobalAveragePooling2D()
        self._dense1 = tf.keras.layers.Dense(256, name='Dense1')

        initializer = tf.keras.initializers.RandomUniform(
            minval=-0.003, maxval=0.003)

        self._action_projection_layer = tf.keras.layers.Dense(
            flat_action_spec[0].shape.num_elements(),
            activation=tf.keras.activations.tanh,
            kernel_initializer=initializer,
            name='action')

    def call(self, observations, step_type=(), network_state=()):
        # We use batch_squash here in case the observations have a time sequence component.
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(batch_squash.flatten, observations)

        # Forward pass through the custom tf layers here (defined above):
        x = self._preprocessing(observations)
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._flatten(x)
        x = self._dense1(x)
        actions = self._action_projection_layer(x)

        actions = common_utils.scale_to_spec(actions, self._single_action_spec)
        actions = batch_squash.unflatten(actions)
        return tf.nest.pack_sequence_as(self._action_spec, [actions]), network_state
    
    
    

class CriticNetworkCustom(network.Network):

    def __init__(self,
                observation_spec,
                action_spec,
                name='CriticNetworkCustom'):
        # Invoke constructor of network.Network
        super(CriticNetworkCustom, self).__init__(
              input_tensor_spec=(observation_spec, action_spec), state_spec=(), name=name)

        self._obs_spec = observation_spec
        self._action_spec = action_spec

        # Encoding layer concatenates state and action inputs, adds dense layer:
        kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=1./3., mode='fan_in', distribution='uniform')
        
        # images
        self._preprocessing = tf.keras.layers.Lambda(lambda x: (tf.cast(x,np.float32) / 255))
        self._conv1 = keras.layers.Conv2D(filters=16, kernel_size=3,strides=2, padding="same", activation="relu")
        self._conv2 = keras.layers.Conv2D(filters=32, kernel_size=3,strides=2, padding="same", activation="relu")
        self._conv3 = keras.layers.Conv2D(filters=64, kernel_size=3,strides=2,padding="same", activation="relu")
        #self._flatten = keras.layers.Flatten()
        self._flatten = keras.layers.GlobalAveragePooling2D()
        
        
        self._combiner = tf.keras.layers.Concatenate(axis=-1)

        # Initialize the custom tf layers here:
        self._dense1 = tf.keras.layers.Dense(64, name='Dense1')
        self._value_layer = tf.keras.layers.Dense(1,
                                                  activation=tf.keras.activations.linear,
                                                  name='Value') # Q-function output

    def call(self, observations, step_type=(), network_state=()):
        # Forward pass through the custom tf layers here (defined above):
        
        x = self._preprocessing(observations[0])
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._flatten(x)
        
        x = self._combiner([x,observations[1]])
        x = self._dense1(x)
        value = self._value_layer(x)

        return tf.reshape(value,[-1]), network_state


# ## Hyperparameters

# In[3]:


env_name = "MinitaurBulletEnv-v0" # @param {type:"string"}

# Use "num_iterations = 1e6" for better results (2 hrs)
# 1e5 is just so this doesn't take too long (1 hr)
num_iterations =50000 # @param {type:"integer"}

initial_collect_steps = 1000 # @param {type:"integer"}
collect_steps_per_iteration = 1 # @param {type:"integer"}
replay_buffer_capacity = 50000 # @param {type:"integer"}

batch_size = 16 # @param {type:"integer"}

critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-4 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.7 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

log_interval = 5000 # @param {type:"integer"}

num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 10 # @param {type:"integer"}

policy_save_interval = 5000 # @param {type:"integer"}


# In[4]:


models_path  = '/home/pico/uni/romi/scanner-gym_models_v2'
'''train_models = ['207_2d','208_2d','209_2d', '210_2d',
               '211_2d','212_2d','213_2d' ,'214_2d']'''
train_models =  ['208_2d','213_2d_','218_2d']
n_images = 5
continuous = True

env_name = 'ScannerEnv-v2'

collect_env = suite_gym.load(env_name,gym_kwargs={'models_path':models_path, 'train_models':train_models,
                                                   'n_images':n_images, 'continuous':continuous,
                                                   'gt_mode':True,'cube_view':'static'}) 

eval_env = suite_gym.load(env_name,gym_kwargs={'models_path':models_path, 'train_models':train_models,
                                                   'n_images':n_images, 'continuous':continuous,
                                                   'gt_mode':True,'cube_view':'static'}) 

tf_collect_env = tf_py_environment.TFPyEnvironment(collect_env)
tf_eval_env = tf_py_environment.TFPyEnvironment(eval_env)


# In[5]:


print('Observation Spec:')
print(tf_collect_env.time_step_spec().observation)
print('Action Spec:')
print(tf_collect_env.action_spec())


# In[6]:


use_gpu = True #@param {type:"boolean"}

strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)


# All variables and Agents need to be created under `strategy.scope()`

# In[7]:


observation_spec, action_spec, time_step_spec = (
      spec_utils.get_tensor_specs(tf_collect_env))

with strategy.scope():
    critic_net = CriticNetworkCustom(observation_spec, action_spec)


# In[8]:


with strategy.scope():
   actor_net = ActorNetworkCustom(observation_spec,
                                    action_spec)


# In[9]:



# With these networks at hand we can now instantiate the agent.
# 

# In[10]:


with strategy.scope():
  train_step = train_utils.create_train_step()

  tf_agent = sac_agent.SacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.keras.optimizers.Adam(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.keras.optimizers.Adam(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.keras.optimizers.Adam(
            learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        train_step_counter=train_step)

  tf_agent.initialize()


# ## Replay Buffer
# 
# In order to keep track of the data collected from the environment, we will use [Reverb](https://deepmind.com/research/open-source/Reverb), an efficient, extensible, and easy-to-use replay system by Deepmind. It stores experience data collected by the Actors and consumed by the Learner during training.
# 
# In this tutorial, this is less important than `max_size` -- but in a distributed setting with async collection and training, you will probably want to experiment with `rate_limiters.SampleToInsertRatio`, using a samples_per_insert somewhere between 2 and 1000. For example:
# ```
# rate_limiter=reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=3.0, min_size_to_sample=3, error_buffer=3.0)
# ```
# 
# 
# 

# In[11]:


table_name = 'uniform_table'
table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1))

reverb_server = reverb.Server([table])


# The replay buffer is constructed using specs describing the tensors that are to be stored, which can be obtained from the agent using `tf_agent.collect_data_spec`.
# 
# Since the SAC Agent needs both the current and next observation to compute the loss, we set `sequence_length=2`.

# In[12]:


reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    sequence_length=2,
    table_name=table_name,
    local_server=reverb_server)


# Now we generate a TensorFlow dataset from the Reverb replay buffer. We will pass this to the Learner to sample experiences for training.

# In[13]:


dataset = reverb_replay.as_dataset(
      sample_batch_size=batch_size, num_steps=2).prefetch(50)
experience_dataset_fn = lambda: dataset


# ## Policies
# 
# In TF-Agents, policies represent the standard notion of policies in RL: given a `time_step` produce an action or a distribution over actions. The main method is `policy_step = policy.step(time_step)` where `policy_step` is a named tuple `PolicyStep(action, state, info)`.  The `policy_step.action` is the `action` to be applied to the environment, `state` represents the state for stateful (RNN) policies and `info` may contain auxiliary information such as log probabilities of the actions.
# 
# Agents contain two policies:
# 
# -   `agent.policy` — The main policy that is used for evaluation and deployment.
# -   `agent.collect_policy` — A second policy that is used for data collection.

# In[14]:


tf_eval_policy = tf_agent.policy
eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
  tf_eval_policy, use_tf_function=True)


# In[15]:


tf_collect_policy = tf_agent.collect_policy
collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
  tf_collect_policy, use_tf_function=True)


# Policies can be created independently of agents. For example, use `tf_agents.policies.random_py_policy` to create a policy which will randomly select an action for each time_step.

# In[16]:


random_policy = random_py_policy.RandomPyPolicy(
  collect_env.time_step_spec(), collect_env.action_spec())


# ## Actors
# The actor manages interactions between a policy and an environment.
#   * The Actor components contain an instance of the environment (as `py_environment`) and a copy of the policy variables.
#   * Each Actor worker runs a sequence of data collection steps given the local values of the policy variables.
#   * Variable updates are done explicitly using the variable container client instance in the training script before calling `actor.run()`.
#   * The observed experience is written into the replay buffer in each data collection step.

# As the Actors run data collection steps, they pass trajectories of (state, action, reward) to the observer, which caches and writes them to the Reverb replay system. 
# 
# We're storing trajectories for frames [(t0,t1) (t1,t2) (t2,t3), ...] because `stride_length=1`.

# In[17]:


rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  reverb_replay.py_client,
  table_name,
  sequence_length=2,
  stride_length=1)


# We create an Actor with the random policy and collect experiences to seed the replay buffer with.

# In[18]:


initial_collect_actor = actor.Actor(
  collect_env,
  random_policy,
  train_step,
  steps_per_run=initial_collect_steps,
  observers=[rb_observer])
initial_collect_actor.run()




# Instantiate an Actor with the collect policy to gather more experiences during training.

# In[19]:


env_step_metric = py_metrics.EnvironmentSteps()
collect_actor = actor.Actor(
  collect_env,
  collect_policy,
  train_step,
  steps_per_run=1,
  metrics=actor.collect_metrics(10),
  summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
  observers=[rb_observer, env_step_metric] )


# Create an Actor which will be used to evaluate the policy during training. We pass in `actor.eval_metrics(num_eval_episodes)` to log metrics later.

# In[20]:


eval_actor = actor.Actor(
  eval_env,
  eval_policy,
  train_step,
  episodes_per_run=num_eval_episodes,
  metrics=actor.eval_metrics(num_eval_episodes),
  summary_dir=os.path.join(tempdir, 'eval'),
)


# ## Learners
# The Learner component contains the agent and performs gradient step updates to the policy variables using experience data from the replay buffer. After one or more training steps, the Learner can push a new set of variable values to the variable container.

# In[21]:


saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

# Triggers to save the agent's policy checkpoints.
learning_triggers = [
    triggers.PolicySavedModelTrigger(
        saved_model_dir,
        tf_agent,
        train_step,
        interval=policy_save_interval),
    triggers.StepPerSecondLogTrigger(train_step, interval=1000),
]

agent_learner = learner.Learner(
  tempdir,
  train_step,
  tf_agent,
  experience_dataset_fn,
  triggers=learning_triggers,
  strategy=strategy)


# ## Metrics and Evaluation
# 
# We instantiated the eval Actor with `actor.eval_metrics` above, which creates most commonly used metrics during policy evaluation:
# * Average return. The return is the sum of rewards obtained while running a policy in an environment for an episode, and we usually average this over a few episodes.
# * Average episode length.
# 
# We run the Actor to generate these metrics.

# In[22]:


def get_eval_metrics():
  eval_actor.run()
  results = {}
  for metric in eval_actor.metrics:
    results[metric.name] = metric.result()
  return results

metrics = get_eval_metrics()


# In[23]:


def log_eval_metrics(step, metrics):
  eval_results = (', ').join(
      '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
  print('step = {0}: {1}'.format(step, eval_results))

log_eval_metrics(0, metrics)


# Check out the [metrics module](https://github.com/tensorflow/agents/blob/master/tf_agents/metrics/tf_metrics.py) for other standard implementations of different metrics.

# ## Training the agent
# 
# The training loop involves both collecting data from the environment and optimizing the agent's networks. Along the way, we will occasionally evaluate the agent's policy to see how we are doing.

# In[24]:


# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = get_eval_metrics()["AverageReturn"]
returns = [avg_return]

for _ in range(num_iterations):
  # Training.
  collect_actor.run()
  loss_info = agent_learner.run(iterations=1)

  # Evaluating.
  step = agent_learner.train_step_numpy

  if eval_interval and step % eval_interval == 0:
    metrics = get_eval_metrics()
    log_eval_metrics(step, metrics)
    returns.append(metrics["AverageReturn"])

   
    '''with train_summary_writer.as_default():
      #plot metrics
      for train_metric in training_metrics:
        train_metric.tf_summaries(train_step=tf.cast(train_step,tf.int64), step_metrics=training_metrics[:])
        train_summary_writer.flush()'''

  if log_interval and step % log_interval == 0:
    print('\rstep = {0}: loss = {1}'.format(step, loss_info.loss.numpy()), end="")

rb_observer.close()
reverb_server.stop()


# ## Visualization
# 

# ### Plots
# 
# We can plot average return vs global steps to see the performance of our agent. In `Minitaur`, the reward function is based on how far the minitaur walks in 1000 steps and penalizes the energy expenditure.

# In[ ]:


#@test {"skip": true}


#steps = range(0, num_iterations + 1, eval_interval)
#plt.plot(steps, returns)
#plt.ylabel('Average Return')
#plt.xlabel('Step')
#plt.ylim()


# In[ ]:





# In[ ]:

test_models = ['208_2d','212_2d','213_2d_','218_2d']
test_data = "tests_.json"
policy_test_sac.test_policy(environment='ScannerEnv-v2', models_path=models_path,
                        models=test_models, policy=eval_policy,
                        n_images=n_images, n_episodes = 18, dest_path=test_data )
stest = policy_test_sac.run_episode(eval_env, tf_eval_env,tf_eval_policy)


# In[ ]:


stest


# In[ ]:




