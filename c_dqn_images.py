#!/usr/bin/env python
# coding: utf-8

# In[1]:

#get_ipython().run_line_magic('reload_ext', 'tensorboard')
import numpy as np
import json
import tensorflow as tf
import os

import time
#get_ipython().run_line_magic('matplotlib', 'inline')
#import matplotlib
#import matplotlib.pyplot as plt
import gym

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

import tf_agents
from tf_agents.networks import categorical_q_network
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
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
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils.common import function, element_wise_squared_loss
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.policies import policy_saver
import logging

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
import csv

seed=10
tf.random.set_seed(seed)
np.random.seed(seed)

from utils import policy_test


# In[2]:


current_path = os.getcwd()
params_file = os.path.join(current_path, 'params.json') 
pm=json.load(open(params_file))
run_label = '08-12-13-18_img_multi05im'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
data_log_path = os.path.join(current_path, 'generated_data/') 

#save parameters and code used for this training
with open(os.path.join(data_log_path,"parameters", run_label+'.json'), 'w') as json_file:
  json.dump(pm, json_file)

src = os.path.join(current_path,"c_dqn_images.py")
dst = os.path.join(data_log_path,"train_code", run_label+'.py')
shutil.copyfile(src, dst)

src = os.path.join(current_path,"scan_gym/scan_gym/envs/ScannerEnv2/scanner_env.py")
dst = os.path.join(data_log_path,"environment_code", run_label+'.py')
shutil.copyfile(src, dst)

# In[3]:


models_path  = '/home/pico/uni/romi/scanner-gym_models_v3'
#models = ['207_2d','208_2d','209_2d','210_2d','211_2d']
'''train_models = ['207_2d','208_2d','209_2d', '210_2d',
               '211_2d','212_2d','213_2d' ,'214_2d']'''
#train_models = ['208_2d','209_2d', '212_2d','213_2d','217_2d','218_2d']
train_models = ['208_2d','212_2d','213_2d_','218_2d']
n_images = 5
continuous = False

#scan_env = gym.make('ScannerEnv-v1', models_path=models_path, train_models=models,
#                   n_images = n_images, continuous=continuous, gt_mode=True, cube_view='static')

env = suite_gym.load('ScannerEnv-v2',gym_kwargs={'models_path':models_path, 'train_models':train_models,
                                                   'n_images':n_images, 'continuous':continuous,
                                                   'gt_mode':True,'cube_view':'static'}) 

tf_env = tf_py_environment.TFPyEnvironment(env)

# In[4]:


tf_env.observation_spec()


# In[5]:


tf_env.action_spec()


# In[6]:


def image_layers():
    input_im = keras.layers.Input(shape=(84,84,3))
    preprocessing = keras.layers.Reshape((84,84,3,1))(input_im)
    #input_vol = keras.layers.Input(shape=(128,128,128))
    #preprocessing = keras.layers.Reshape((128,128,128,1))(input_vol)
    preprocessing = keras.layers.Lambda(lambda x: (tf.cast(x,np.float32) / 255))(preprocessing) #normalize 0-1
    
    stride = 2
    
    x = keras.layers.Conv2D(filters=16, kernel_size=3,strides=stride, padding="same", activation="relu")(preprocessing)
    #x = keras.layers.MaxPool3D(pool_size=2)(x)
    #x = keras.layers.BatchNormalization()(x)
    
    x = keras.layers.Conv2D(filters=32, kernel_size=3,strides=stride, padding="same", activation="relu")(x)
    #x = keras.layers.MaxPool3D(pool_size=2)(x)
    #x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(filters=64, kernel_size=3,strides=stride,padding="same", activation="relu")(x)
    #x = keras.layers.MaxPool3D(pool_size=2)(x)
    #x = keras.layers.BatchNormalization()(x)
    
    #x = keras.layers.Conv3D(filters=64, kernel_size=3,strides=stride,padding="same", activation="relu")(x)
    #x = keras.layers.MaxPool3D(pool_size=2)(x)
   
    x = keras.layers.Flatten()(x)
    #x = keras.layers.GlobalAveragePooling3D()(x)
  
    #x = keras.layers.Dense(512)(x)
                                        
    model = keras.models.Model(inputs=input_im,outputs=x)
    model.summary()
    return model
    

#scale range 0 to 1
oldmin = tf_env.observation_spec()[1].minimum
oldmax = tf_env.observation_spec()[1].maximum

#oldmin = tf_env.observation_spec().minimum
#oldmax = tf_env.observation_spec().maximum


print(oldmin,oldmax)
    
def input_vect_layers():
    input_ = keras.layers.Input(shape=(2,))
    preprocessing = keras.layers.Lambda(lambda x: ((x-oldmin)*(1.- 0.)/(oldmax-oldmin)) + 0. )(input_)
    #x = keras.layers.Dense(32)(preprocessing)
    return keras.models.Model(inputs=input_,outputs=preprocessing)


# In[7]:


#network
#preprocessing_layers=image_layers()
preprocessing_layers=(image_layers(),input_vect_layers())
#preprocessing_layers=input_vect_layers()

preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
dense_l = pm['model']['fc_layer_params']
if len(dense_l) == 1:
    fc_layer_params = (dense_l[0],)
else:
    fc_layer_params = dense_l


categorical_q_net = categorical_q_network.CategoricalQNetwork(
tf_env.observation_spec(),
tf_env.action_spec(),
preprocessing_layers=preprocessing_layers,
preprocessing_combiner=preprocessing_combiner,
fc_layer_params=fc_layer_params,
num_atoms=pm['categorical_dqn']['n_atoms'])


# In[8]:


#agent
train_step = tf.Variable(0)
#optimizer = keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0,
#            epsilon=0.00001, centered=True)
'''lr_decay =  keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0.005, # initial ε
            decay_steps = pm['agent']['decay_steps'], 
            end_learning_rate=0.0005)
optimizer = keras.optimizers.Adam(learning_rate=lambda: lr_decay(train_step))'''


optimizer = keras.optimizers.Adam(learning_rate=pm['model']['learning_rate'])

epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=1.0, # initial ε
            decay_steps = pm['agent']['decay_steps'], 
            end_learning_rate=0.05) # final ε

agent = categorical_dqn_agent.CategoricalDqnAgent(tf_env.time_step_spec(),
                tf_env.action_spec(),
                categorical_q_network=categorical_q_net,
                optimizer=optimizer,
                min_q_value=pm['categorical_dqn']['min_q_value'],
                max_q_value=pm['categorical_dqn']['max_q_value'],
                target_update_period=pm['agent']['target_update_period'],
                td_errors_loss_fn=element_wise_squared_loss,#keras.losses.Huber(reduction="none"),#element_wise_squared_loss,
                gamma=pm['agent']['gamma'], # discount factor
                train_step_counter=train_step,
                n_step_update =  pm['categorical_dqn']['n_step_update'],                                
                epsilon_greedy=lambda: epsilon_fn(train_step))
agent.initialize()


# In[9]:


#Replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec= agent.collect_data_spec,
    batch_size= tf_env.batch_size,
    max_length=pm['rbuffer']['max_length'])


# In[10]:


#observer
#observer is just a function (or a callable object) that takes a trajectory argument,
#add_method() method (bound to the replay_buffer object) can be used as observer
replay_buffer_observer = replay_buffer.add_batch


# In[11]:


#observer for training metrics
training_metrics = [
tf_metrics.NumberOfEpisodes(),
tf_metrics.AverageEpisodeLengthMetric(),
tf_metrics.EnvironmentSteps(),
tf_metrics.AverageReturnMetric(),
tf_metrics.MaxReturnMetric(),
tf_metrics.MinReturnMetric(),   
]


# In[12]:


#custom observer
class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


# In[13]:


#Collect Driver
update_period = pm['collect_driver']['num_steps'] # train the model every x steps
collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + training_metrics,
    num_steps=update_period) # collect x steps for each training iteration

#+ training_metrics,


# In[14]:


# random policy driver to start filling the buffer
random_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                        tf_env.action_spec())

ns = pm['rnd_policy']['num_steps']
init_driver = DynamicStepDriver(
            tf_env,
            random_collect_policy,
            observers=[replay_buffer_observer, ShowProgress(ns)],
            num_steps=ns)
#],
final_time_step, final_policy_state = init_driver.run()


# In[15]:


#use buffer as tf API dataset ()
dataset = replay_buffer.as_dataset(
        sample_batch_size=pm['rbuffer']['sample_batch_size'],
        num_steps=pm['categorical_dqn']['n_step_update'] + 1,
        num_parallel_calls=3).prefetch(3)


# In[16]:


#convert main functions to tensorflow functions to speed up training
collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)


# In[17]:


#@tf.function
def train_agent(n_iterations):
    #reset metrics
    for m in training_metrics:
        m.reset()
    time_step = None
    policy_state = ()#agent.collect_policy.get_initial_state(tf_env.batch_size)
    agent.train_step_counter.assign(0)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f} epsilon{}".format(iteration, train_loss.loss.numpy(),epsilon_fn(train_step)), end="")
        if iteration % 100 == 0:
            with train_summary_writer.as_default():
                #plot metrics
                for train_metric in training_metrics:
                    train_metric.tf_summaries(train_step=tf.cast(agent.train_step_counter,tf.int64), step_metrics=training_metrics[:])
                train_summary_writer.flush()
                
        if iteration % 100 == 0:
            with train_summary_writer.as_default():
                #plot train loss          
                tf.summary.scalar('train_loss', train_loss.loss.numpy(), step=tf.cast(agent.train_step_counter,tf.int64))
                #plot NN weights
                for layer in  categorical_q_net.layers:
                    for weight in layer.weights:
                        tf.summary.histogram(weight.name,weight,step=tf.cast(agent.train_step_counter,tf.int64))
                train_summary_writer.flush()


        if iteration % 5000 == 0:
          test_models =train_models
          policy_test.test_policy(environment='ScannerEnv-v2', models_path=models_path,
                                  models=test_models, policy=agent.policy,
                                  n_images=n_images, n_episodes = 50, dest_path="" )

# In[18]:


train_dir = os.path.join(data_log_path,"logs/",run_label)   
train_summary_writer = tf.summary.create_file_writer(
            train_dir, flush_millis=10000)
#train_summary_writer.set_as_default()


# In[19]:


# Launch TensorBoard with objects in the log directory
# This should launch tensorboard in your browser, but you may not see your metadata.
#%tensorboard --logdir=logs --reload_interval=15


# In[20]:


#tf.summary.scalar('avgreturn', training_metrics[3].result().numpy(), step=tf.cast(agent.train_step_counter,tf.int64))


# In[ ]:


#get_ipython().run_line_magic('time', "train_agent(pm['misc']['n_iterations'])")
train_agent(pm['misc']['n_iterations'])

# In[ ]:


#save model,
policy_dir = os.path.join(data_log_path,"policies", run_label)
tf_policy_saver = policy_saver.PolicySaver(agent.policy)
tf_policy_saver.save(policy_dir)


# In[ ]:


# test learned policy
'''test_models = ['206_2d','207_2d','208_2d','209_2d', '210_2d',
               '211_2d','212_2d','213_2d' ,'214_2d' ,'215_2d',
               '216_2d','217_2d','218_2d']'''
test_models = ['208_2d','212_2d','213_2d_','218_2d']
test_data = os.path.join(data_log_path,"tests", run_label+'.json')
policy_test.test_policy(environment='ScannerEnv-v2', models_path=models_path,
                        models=test_models, policy=agent.policy,
                        n_images=n_images, n_episodes = 180, dest_path=test_data )

stest = policy_test.run_episode(env, tf_env, agent.policy)
for i in stest:
  print(i)
# In[ ]:





# In[ ]:





# In[ ]:




