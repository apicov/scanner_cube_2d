
from scan_gym import envs
from scan_gym.envs.ScannerEnv import space_carving
import gym
import numpy as np
import json
from PIL import Image
from utils import *
import glob
import os
import copy
import time

TOTAL_THETA = 180
TOTAL_PHI = 4


def calculate_position(init_state,steps):
    n_positions = TOTAL_THETA
    n_pos = init_state + steps
    if n_pos>(n_positions-1):
        n_pos -= n_positions
    elif n_pos<0:
        n_pos += n_positions
    return n_pos


def test_rnd_policy(models_path, model, n_images, init_position, theta_bias):
    scan_env = gym.make('ScannerEnv-v2', models_path=models_path, train_models=[model],
                        n_images = n_images, continuous=False, gt_mode=True, cube_view='static')
    
    scan_env.reset(theta_init=init_position[0], phi_init=init_position[1], theta_bias=theta_bias)

    for i in range(1000):
        _, _, done, _ = scan_env.step(np.random.randint(scan_env.nA))
        if done:
            break

    return scan_env.total_reward, scan_env.spc.gt_compare_solid()


def test_uniform(models_path,model,n_images, init_theta):
    total_theta_positions = TOTAL_THETA
    spc = space_carving.space_carving_rotation_2d( os.path.join(models_path, model),
                        gt_mode=True, theta_bias=0,
                        total_theta_positions=total_theta_positions,
                        cube_view='static')
    dist = total_theta_positions//n_images

    theta_count = init_theta
    for i in range(n_images):
        spc.carve(theta_count,0)
        theta_count = calculate_position(theta_count,dist)
        
    gt_sim = spc.gt_compare_solid()
    return gt_sim



models_path  = '/home/pico/uni/romi/scanner-gym_models_v2'
'''models = ['206_2d','207_2d','208_2d','209_2d','210_2d','211_2d','212_2d','213_2d','214_2d','215_2d','216_2d', '217_2d','218_2d']'''
models = ['212_2d','213_2d', '218_2d']
n_images = 10


# generate set of random initial positions and position biases for using in all tests
seed = 43
np.random.seed(seed)
n_init_positions = 180
theta_phi_posbias = np.random.randint((0,0,0),(TOTAL_THETA,TOTAL_PHI,1),(n_init_positions,3))


f = open("uni_rnd_policy_runs_10_v2.json",'w')


data ={}
for m in models:
    m_data = {'rnd':{'cum_reward':[],'gt_dists':[],
                     'cum_reward_mean':0, 'cum_reward_std':0,
                     'gt_mean':0,'gt_std':0},
              
              'uni':{'gt_dists':[],'gt_mean':0,'gt_std':0},

              }
    
    print(m)
    
    #collect data random policy
    for i in theta_phi_posbias:
        time1 = time.time()
        cum_reward, gt_dist = \
            test_rnd_policy(models_path, m, n_images, init_position=i[:2], theta_bias=i[2])

        m_data['rnd']['cum_reward'].append(cum_reward)
        m_data['rnd']['gt_dists'].append(gt_dist)
       
        time2 = time.time()
        print(i, m_data['rnd']['cum_reward'][-1], m_data['rnd']['gt_dists'][-1],time2-time1)


    m_data['rnd']['cum_reward_mean'] = float(np.mean(m_data['rnd']['cum_reward']))
    m_data['rnd']['cum_reward_std'] = float(np.std(m_data['rnd']['cum_reward']))
    m_data['rnd']['gt_mean'] = float(np.mean(m_data['rnd']['gt_dists']))
    m_data['rnd']['gt_std'] = float(np.std(m_data['rnd']['gt_dists']))

    print('cum_reward_mean',m_data['rnd']['cum_reward_mean'],
          'cum_reward_std',m_data['rnd']['cum_reward_std'])
    print('gt_mean',m_data['rnd']['gt_mean'],'gt_std',m_data['rnd']['gt_std'])
    
    
    #collect data uniform distribution policy
    steps = TOTAL_THETA//n_images
    for i in range(steps):
        time1 = time.time()
        gt_dist = test_uniform(models_path, m, n_images, i)
        m_data['uni']['gt_dists'].append(gt_dist)
        time2 = time.time()
        print(i, m_data['uni']['gt_dists'][-1],time2-time1)

    m_data['uni']['gt_mean'] = float(np.mean(m_data['uni']['gt_dists']))
    m_data['uni']['gt_std'] = float(np.std(m_data['uni']['gt_dists']))
    print('gt_mean',m_data['uni']['gt_mean'],'gt_std',m_data['uni']['gt_std'])

    data[m] = copy.deepcopy(m_data)


json.dump(data, f)
f.close()
    

