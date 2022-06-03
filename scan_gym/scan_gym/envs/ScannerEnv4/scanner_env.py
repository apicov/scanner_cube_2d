import numpy as np
#import cv2
from os import listdir
from os.path import isfile, join
import gym
from gym import error, spaces, utils
import glob
from PIL import Image
from skimage.morphology import binary_dilation
import json
from .utils import *
import glob
import os
from .space_carving import *
import random


class ScannerEnv(gym.Env):
    """
    Custom OpenAI Gym environment  for training 3d scannner
    """
    metadata = {'render.modes': ['human']}
    def __init__(self,models_path,train_models,n_images = 10, continuous=False,gt_mode=True,cube_view='dynamic'):
        super(ScannerEnv, self).__init__()
        #self.__version__ = "7.0.1"
        # if gt_mode true, ground truth model is used by space carving object (for comparing it against current volume)
        self.gt_mode = gt_mode
        # number of images that must be collected 
        self.n_images = n_images 
        self.models_path = models_path
        # total of posible positions for theta  in env
        self.theta_n_positions = 360
        # total of posible positions for phi in env
        self.phi_n_positions = 4
        # 3d models used for training
        self.train_models = train_models
        # if static, figure in cube is always seen from the same perspective
        #if dynamic, position figure in cube rotates according to the camera perspective
        self.cube_view = cube_view
        

        #for activating continuous action mode
        self.continuous = continuous

        '''the state returned by this environment consiste of
        last three images,
        the volume being carved , theta position , phi position'''
        # images
        #self.images_shape = (84,84,3)
        #self.img_obs_space = gym.spaces.Box(low=0, high=255, shape=self.images_shape, dtype=np.uint8)
        
        # volume used in the carving
        self.volume_shape = (64,64,64) #(128,128,128)#(64,64,64)
        self.vol_obs_space = gym.spaces.Box(low=-1, high=1, shape=self.volume_shape, dtype=np.float16)

        '''# theta positions                                          
        t_low = np.array([0])
        t_high = np.array([self.theta_n_positions-1])                                           
        self.theta_obs_space = gym.spaces.Box(t_low, t_high, dtype=np.int32)

        # phi positions                                          
        p_low = np.array([0])
        p_high = np.array([self.phi_n_positions-1])                                           
        self.phi_obs_space = gym.spaces.Box(p_low, p_high, dtype=np.int32)'''


        #self.observation_space = gym.spaces.Tuple((self.img_obs_space, self.vol_obs_space, self.theta_obs_space, self.phi_obs_space))
        self.observation_space = self.vol_obs_space

        if self.continuous:
             self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)     
        else:
            # map action with correspondent movements in theta and phi
            # theta numbers are number of steps relative to current position
            # phi numbers are absolute position in phi
            #(theta,phi)

            '''self.actions = {0:(1,0),1:(2,0),2:(5,0),3:(7,0),4:(10,0),5:(15,0),6:(20,0),7:(25,0),8:(30,0),9:(35,0),10:(45,0),
                            11:(1,1),12:(2,1),13:(5,1),14:(7,1),15:(10,1),16:(15,1),17:(20,1),18:(25,1),19:(30,1),20:(35,1),21:(45,1),
                            22:(1,2),23:(2,2),24:(5,2),25:(7,2),26:(10,2),27:(15,2),28:(20,2),29:(25,2),30:(30,2),31:(35,2),32:(45,2),
                            33:(1,3),34:(2,3),35:(5,3),36:(7,3),37:(10,3),38:(15,3),39:(20,3),40:(25,3),41:(30,3),42:(35,3),43:(45,3),
                            44:(50,0),45:(55,0),46:(60,0),47:(65,0),48:(70,0),49:(75,0),50:(80,0),51:(85,0),52:(90,0),
                            53:(50,1),54:(55,1),55:(60,1),56:(65,1),57:(70,1),58:(75,1),59:(80,1),60:(85,1),61:(90,1),
                            62:(50,2),63:(55,2),64:(60,2),65:(65,2),66:(70,2),67:(75,2),68:(80,2),69:(85,2),70:(90,2),
                            71:(50,3),72:(55,3),73:(60,3),74:(65,3),75:(70,3),76:(75,3),77:(80,3),78:(85,3),79:(90,3),
                            80:(95,0),81:(100,0),82:(105,0),83:(110,0),84:(115,0),85:(120,0),86:(125,0),87:(130,0),88:(135,0),
                            89:(95,1),90:(100,1),91:(105,1),92:(110,1),93:(115,1),94:(120,1),95:(125,1),96:(130,1),97:(135,1),
                            98:(95,2),99:(100,2),100:(105,2),101:(110,2),102:(115,2),103:(120,2),104:(125,2),105:(130,2),106:(135,2),
                            107:(95,3),108:(100,3),109:(105,3),110:(110,3),111:(115,3),112:(120,3),113:(125,3),114:(130,3),115:(135,3)}'''

            
            '''self.actions = {0:(2,0),1:(5,0),2:(10,0),3:(15,0),4:(25,0),5:(45,0),
                            6:(2,1),7:(5,1),8:(10,1),9:(15,1),10:(25,1),11:(45,1),
                            12:(2,2),13:(5,2),14:(10,2),15:(15,2),16:(25,2),17:(45,2),
                            18:(2,3),19:(5,3),20:(10,3),21:(15,3),22:(25,3),23:(45,3)}'''


            '''self.actions = {0:(2,0),1:(4,0),2:(6,0),3:(8,0),4:(10,0),5:(12,0),
                            6:(14,0),7:(16,0),8:(18,0),9:(20,0),10:(22,0),11:(24,0),
                            12:(26,0),13:(28,0),14:(30,0),15:(32,0),16:(34,0),17:(36,0),
                            18:(38,0),19:(40,0),20:(42,0),21:(44,0),22:(46,3),23:(48,0),
                            24:(1,0)}'''

            self.actions = {0:(2,0),1:(4,0),2:(6,0),3:(8,0),4:(10,0),5:(12,0),
                            6:(14,0),7:(16,0),8:(18,0),9:(20,0),10:(22,0),11:(24,0),
                            12:(26,0),13:(28,0),14:(30,0),15:(32,0),16:(34,0),17:(36,0),
                            18:(38,0),19:(40,0),20:(42,0),21:(44,0),22:(46,3),23:(48,0),
                            24:(50,0),25:(52,0),26:(54,0),27:(56,0),28:(58,3),29:(60,0),
                            30:(62,0),31:(64,0),32:(66,0),33:(68,0),34:(70,3),35:(72,0),
                            36:(74,0),37:(1,0)}
   
            

            self.action_space = gym.spaces.Discrete(len(self.actions))

        self.zeros = np.zeros((64,64,64))
        #self._spec.id = "Romi-v0"
        self.reset()

    def reset(self,theta_init=-1,phi_init=-1,theta_bias=0):
        self.num_steps = 0
        self.total_reward = 0
        self.done = False

        # keep track of visited positions during the episode
        self.visited_positions = [] 

        #inital position of the camera, if -1 choose random
        self.init_theta = theta_init
        self.init_phi = phi_init
        
        if self.init_theta == -1:
            self.init_theta = np.random.randint(0,self.theta_n_positions)
            self.current_theta = self.init_theta
        else:
            self.current_theta = self.init_theta

        if self.init_phi == -1:
            self.init_phi = np.random.randint(0,self.phi_n_positions)
            self.current_phi = self.init_phi 
        else:
            self.current_phi = self.init_phi

        # simulates rotation of object (z axis) by n steps (for data augmentation), -1 for random rotation
        if theta_bias == -1: 
            self.theta_bias = np.random.randint(0,self.theta_n_positions)
        else:
            self.theta_bias = theta_bias

        #append initial position to visited positions list    
        self.visited_positions.append((self.current_theta, self.current_phi))
    
                
        # take random  model from available models list
        model = random.choice(self.train_models)

        # create space carving object
        self.spc = space_carving_rotation_2d( os.path.join(self.models_path, model),
                        gt_mode=self.gt_mode, theta_bias=self.theta_bias,
                        total_theta_positions=self.theta_n_positions,
                        cube_view=self.cube_view)

        # carve image from initial position
        self.spc.carve(self.current_theta, self.current_phi)
        vol = self.spc.volume

        # count of empty,undetermined and solid voxels in volume
        # -1's (empty space), 0's (undetermined) and 1's (solid) from 3d volume
        #last count of empty spaces
        self.last_empty_voxel_count =  np.count_nonzero(vol == -1) 


        #self.current_state = ( vol.astype('float16'), np.array([self.current_theta, self.current_phi],dtype=int))
        self.current_state = vol.astype('float16')  #self.zeros #vol.astype('float16')

        return self.current_state


  

    def step(self, action):
        self.num_steps += 1

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
            # decode theta and phi
            theta = self.theta_from_continuous(action[0])
            phi = self.phi_from_continuous(action[1])

        else:
            # decode theta and phi from action number
            theta = self.actions[action][0]
            phi = self.actions[action][1]

            
        #move n theta steps from current theta position
        self.current_theta = self.calculate_theta_position(self.current_theta, theta)
        # phi indicates absolute position
        #check phi limits
        if phi < 0:
            phi = 0
        elif phi >= self.phi_n_positions:
            phi = self.phi_n_positions-1
            
        self.current_phi = phi
    
        
        self.visited_positions.append((self.current_theta, self.current_phi))

        #carve in new position
        self.spc.carve(self.current_theta, self.current_phi)
        vol = self.spc.volume

        #count empty voxels of current volume
        self.current_empty_voxel_count = np.count_nonzero(vol == -1) 
        delta_empty_voxels = self.current_empty_voxel_count - self.last_empty_voxel_count
        self.last_empty_voxel_count = self.current_empty_voxel_count
        reward = (delta_empty_voxels / self.spc.gt_n_empty_voxels)

        
        if self.num_steps >= (self.n_images-1):
            self.done = True
           
        self.total_reward += reward


        '''self.current_state = (self.im3,
                              vol.astype('float16') ,
                              np.array([self.current_theta],dtype=int),
                              np.array([self.current_phi],dtype=int))'''
        #self.current_state = ( vol.astype('float16'), np.array([self.current_theta, self.current_phi],dtype=int))
        self.current_state = vol.astype('float16')  #self.zeros #vol.astype('float16')

        return self.current_state, reward, self.done, {}

 
    def minMaxNorm(self,old, oldmin, oldmax , newmin , newmax):
        return ( (old-oldmin)*(newmax-newmin)/(oldmax-oldmin) ) + newmin


    def theta_from_continuous(self, cont):
        '''converts float from (-1,1) to int (-90,90) '''
        return int(self.minMaxNorm(cont,-1.0,+1.0,-90,90))

    def phi_from_continuous(self, cont):
        '''converts float from (-1,1) to int (0,3) '''
        return int(self.minMaxNorm(cont,-1.0,+1.0,0.0,3.0))

   
    def calculate_theta_position(self,curr_theta,steps):
        n_pos = curr_theta + steps
        if n_pos>(self.theta_n_positions-1):
            n_pos -= self.theta_n_positions
        elif n_pos<0:
            n_pos += self.theta_n_positions
        return n_pos


    @property
    def nA(self):
        return self.action_space.n

    def render(self, mode='human', close=False):
        """
        :param mode:
        :return:
        """
        return
