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
    def __init__(self,models_path,train_models,n_images = 10, continuous=False,gt_mode=True,cube_view='dynamic',multi_input=False):
        super(ScannerEnv, self).__init__()
        #self.__version__ = "7.0.1"
        # if gt_mode true, ground truth model is used by space carving object (for comparing it against current volume)
        self.gt_mode = gt_mode
        # number of images that must be collected 
        self.n_images = n_images 
        self.models_path = models_path
        # total of posible positions for theta  in env
        self.theta_n_positions = 180
        # total of posible positions for phi in env
        self.phi_n_positions = 4
        # 3d models used for training
        self.train_models = train_models
        # if static, figure in cube is always seen from the same perspective
        #if dynamic, position figure in cube rotates according to the camera perspective
        self.cube_view = cube_view

        # keep empty spaces measurements
        self.es_similarity = []
        self.solid_similarities = []
        

        #for activating continuous action mode
        self.continuous = continuous

        self.multi_in = multi_input

        '''the state returned by this environment consiste of
        last three images,
        the volume being carved , theta position , phi position'''
        # images
        self.images_shape = (84,84,3)
        self.img_obs_space = gym.spaces.Box(low=0, high=255, shape=self.images_shape, dtype=np.uint8)
        
        # volume used in the carving
        #self.volume_shape = (64,64,64) #(128,128,128)#(64,64,64) np.float16
        #self.vol_obs_space = gym.spaces.Box(low=-1, high=1, shape=self.volume_shape, dtype=np.float16)

        '''# theta positions                                          
        t_low = np.array([0])
        t_high = np.array([self.theta_n_positions-1])                                           
        self.theta_obs_space = gym.spaces.Box(t_low, t_high, dtype=np.int32)

        # phi positions                                          
        p_low = np.array([0])
        p_high = np.array([self.phi_n_positions-1])                                           
        self.phi_obs_space = gym.spaces.Box(p_low, p_high, dtype=np.int32)'''

        # theta and phi positions
        lowl = np.array([0,0])
        highl = np.array([self.theta_n_positions-1, self.phi_n_positions-1])
        self.vec_ob_space = gym.spaces.Box(lowl, highl, dtype=np.int32)

        # theta and phi positions
        #lowl = np.array([0]*6)
        #highl = np.array([self.theta_n_positions-1]*3 + [self.phi_n_positions-1]*3)
        #self.vec_ob_space = gym.spaces.Box(lowl, highl, dtype=np.int32)

        '''lowl = np.array([-1]*self.n_images)
        highl = np.array([179]*self.n_images)                                           
        self.vec_ob_space = gym.spaces.Box(lowl, highl, dtype=np.int32)'''


        #self.observation_space = gym.spaces.Tuple((self.img_obs_space, self.vol_obs_space, self.theta_obs_space, self.phi_obs_space))
        #self.observation_space = gym.spaces.Tuple((self.vol_obs_space,self.vec_ob_space))

        if self.multi_in:
            self.observation_space = gym.spaces.Tuple((self.img_obs_space,self.vec_ob_space))
        else:
            self.observation_space = self.img_obs_space
            #self.observation_space = self.vec_ob_space

        if self.continuous:
             self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)     
        else:
            # map action with correspondent movements in theta and phi
            # theta numbers are number of steps relative to current position
            # phi numbers are absolute position in phi
            #(theta,phi)

           

            if self.multi_in:
                self.actions = {0 : (0,0), 1 : (0,1), 2 : (0,2), 3 : (0,3), 4 : (2,0), 
                                5 : (2,1), 6 : (2,2), 7 : (2,3), 8 : (4,0), 9 : (4,1), 
                                10 : (4,2), 11 : (4,3), 12 : (7,0), 13 : (7,1), 14 : (7,2), 
                                15 : (7,3), 16 : (9,0), 17 : (9,1), 18 : (9,2), 19 : (9,3), 
                                20 : (11,0), 21 : (11,1), 22 : (11,2), 23 : (11,3), 24 : (14,0), 
                                25 : (14,1), 26 : (14,2), 27 : (14,3), 28 : (16,0), 29 : (16,1), 
                                30 : (16,2), 31 : (16,3), 32 : (18,0), 33 : (18,1), 34 : (18,2), 
                                35 : (18,3), 36 : (21,0), 37 : (21,1), 38 : (21,2), 39 : (21,3), 
                                40 : (23,0), 41 : (23,1), 42 : (23,2), 43 : (23,3), 44 : (26,0), 
                                45 : (26,1), 46 : (26,2), 47 : (26,3), 48 : (28,0), 49 : (28,1), 
                                50 : (28,2), 51 : (28,3), 52 : (30,0), 53 : (30,1), 54 : (30,2), 
                                55 : (30,3), 56 : (33,0), 57 : (33,1), 58 : (33,2), 59 : (33,3), 
                                60 : (35,0), 61 : (35,1), 62 : (35,2), 63 : (35,3), 64 : (37,0), 
                                65 : (37,1), 66 : (37,2), 67 : (37,3), 68 : (40,0), 69 : (40,1), 
                                70 : (40,2), 71 : (40,3), 72 : (42,0), 73 : (42,1), 74 : (42,2), 
                                75 : (42,3), 76 : (45,0), 77 : (45,1), 78 : (45,2), 79 : (45,3), 
                                80 : (67,0), 81 : (67,1), 82 : (67,2), 83 : (67,3), 84 : (90,0), 
                                85 : (90,1), 86 : (90,2), 87 : (90,3), }
            else:
                self.actions = {0 : (0,-3), 1 : (0,-2), 2 : (0,-1), 3 : (0,0), 4 : (0,1), 
                                5 : (0,2), 6 : (0,3), 7 : (2,-3), 8 : (2,-2), 9 : (2,-1), 
                                10 : (2,0), 11 : (2,1), 12 : (2,2), 13 : (2,3), 14 : (4,-3), 
                                15 : (4,-2), 16 : (4,-1), 17 : (4,0), 18 : (4,1), 19 : (4,2), 
                                20 : (4,3), 21 : (7,-3), 22 : (7,-2), 23 : (7,-1), 24 : (7,0), 
                                25 : (7,1), 26 : (7,2), 27 : (7,3), 28 : (9,-3), 29 : (9,-2), 
                                30 : (9,-1), 31 : (9,0), 32 : (9,1), 33 : (9,2), 34 : (9,3), 
                                35 : (11,-3), 36 : (11,-2), 37 : (11,-1), 38 : (11,0), 39 : (11,1), 
                                40 : (11,2), 41 : (11,3), 42 : (14,-3), 43 : (14,-2), 44 : (14,-1), 
                                45 : (14,0), 46 : (14,1), 47 : (14,2), 48 : (14,3), 49 : (16,-3), 
                                50 : (16,-2), 51 : (16,-1), 52 : (16,0), 53 : (16,1), 54 : (16,2), 
                                55 : (16,3), 56 : (18,-3), 57 : (18,-2), 58 : (18,-1), 59 : (18,0), 
                                60 : (18,1), 61 : (18,2), 62 : (18,3), 63 : (21,-3), 64 : (21,-2), 
                                65 : (21,-1), 66 : (21,0), 67 : (21,1), 68 : (21,2), 69 : (21,3), 
                                70 : (23,-3), 71 : (23,-2), 72 : (23,-1), 73 : (23,0), 74 : (23,1), 
                                75 : (23,2), 76 : (23,3), 77 : (26,-3), 78 : (26,-2), 79 : (26,-1), 
                                80 : (26,0), 81 : (26,1), 82 : (26,2), 83 : (26,3), 84 : (28,-3), 
                                85 : (28,-2), 86 : (28,-1), 87 : (28,0), 88 : (28,1), 89 : (28,2), 
                                90 : (28,3), 91 : (30,-3), 92 : (30,-2), 93 : (30,-1), 94 : (30,0), 
                                95 : (30,1), 96 : (30,2), 97 : (30,3), 98 : (33,-3), 99 : (33,-2), 
                                100 : (33,-1), 101 : (33,0), 102 : (33,1), 103 : (33,2), 104 : (33,3), 
                                105 : (35,-3), 106 : (35,-2), 107 : (35,-1), 108 : (35,0), 109 : (35,1), 
                                110 : (35,2), 111 : (35,3), 112 : (37,-3), 113 : (37,-2), 114 : (37,-1), 
                                115 : (37,0), 116 : (37,1), 117 : (37,2), 118 : (37,3), 119 : (40,-3), 
                                120 : (40,-2), 121 : (40,-1), 122 : (40,0), 123 : (40,1), 124 : (40,2), 
                                125 : (40,3), 126 : (42,-3), 127 : (42,-2), 128 : (42,-1), 129 : (42,0), 
                                130 : (42,1), 131 : (42,2), 132 : (42,3), 133 : (45,-3), 134 : (45,-2), 
                                135 : (45,-1), 136 : (45,0), 137 : (45,1), 138 : (45,2), 139 : (45,3)} 
           
           

            self.action_space = gym.spaces.Discrete(len(self.actions))

        self.zeros = np.zeros((84,84,3))
        #self._spec.id = "Romi-v0"
        self.reset()
        
        
    def reset(self,theta_init=-1,phi_init=0,theta_bias=0):
        self.gano = False
        self.num_steps = 0
        self.total_reward = 0
        self.done = False
        self.solid_similarities = []

        #keep camera positions of the episode
        self.theta_history = []
        self.phi_history = []

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
        #self.last_empty_voxel_count =  np.count_nonzero(vol == -1)
        #self.es_similarity.append( self.spc.gt_compare_empty_voxels())
        self.solid_similarities.append(self.spc.gt_compare_solid())

        # get camera image
        im = np.array(self.spc.get_image(self.current_theta, self.current_phi))
        # need 3 last images, this is first taken image so copy it 3 times
        # and adjust dimensions (height,width,channel)
        self.im3 = np.array([im,im,im]).transpose(1,2,0)

        #keep camera positions of the episode
        self.theta_history.append(self.current_theta)
        self.phi_history.append( self.current_phi)

        #last 3 theta and phi values
        theta_state = [self.current_theta]*3
        phi_state = [self.current_phi]*3
        
           

        '''if self.gt_mode is True:
            # keep similarity ratio of current volume and groundtruth volume
            # for calculating deltas of similarity ratios in next steps
            self.last_gt_ratio = self.spc.gt_compare_solid()
        else:
            #get number of -1's (empty space), 0's (undetermined) and 1's (solid) from 3d volume
            self.voxel_count = [np.count_nonzero(vol == -1),
                                np.count_nonzero(vol == 0),
                                np.count_nonzero(vol == 1) ] 
            self.last_voxel_count = self.voxel_count.copy() '''




            
        '''self.current_state = (self.im3,
                              vol.astype('float16') ,
                              np.array([self.current_theta],dtype=int),
                              np.array([self.current_phi],dtype=int))'''

        #self.current_state = ( vol.astype('float16'), np.array([self.current_theta, self.current_phi],dtype=int))

        #self.current_state = ( vol.astype('float16'), np.array(theta_state+phi_state,dtype=int))

        if self.multi_in:
            self.current_state = (self.im3, np.array([self.current_theta, self.current_phi],dtype=int))
            #self.current_state = (self.im3, np.array(theta_state+phi_state,dtype=int))
            #self.current_state = ( vol.astype('float16'), np.array([self.current_theta, self.current_phi],dtype=int))
        else:
            self.current_state =self.im3 #self.zeros #self.im3

            #self.current_state =  np.array([self.current_theta, self.current_phi],dtype=int)
            #self.current_state =  np.array([0,0],dtype=int)
            #self.current_state =  np.array(theta_state+phi_state,dtype=int)

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

        if  not self.multi_in:
            # move phi
            phi +=  self.current_phi
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
        #self.current_empty_voxel_count = np.count_nonzero(vol == -1) 

        # get camera image
        im = np.array(self.spc.get_image(self.current_theta, self.current_phi))
        # need 3 last images, #and adjust dimensions (height,width,channel)
        self.im3 = np.array([self.im3[:,:,1],self.im3[:,:,2], im]).transpose(1,2,0)

        #keep camera positions of the episode
        self.theta_history.append(self.current_theta)
        self.phi_history.append( self.current_phi)

        #last 3 theta and phi values
        if len(self.theta_history) < 3:
            #it means thhere are 2 elements in history
            theta_state = [self.theta_history[0],self.theta_history[0],self.theta_history[1]]
            phi_state = [self.phi_history[0],self.phi_history[0],self.phi_history[1]]
        else:
            theta_state = self.theta_history[-3:]
            phi_state = self.phi_history[-3:]



        #delta_empty_voxels = self.current_empty_voxel_count - self.last_empty_voxel_count
        #self.last_empty_voxel_count = self.current_empty_voxel_count
        #reward = (delta_empty_voxels / self.spc.gt_n_empty_voxels) * self.num_steps

        #reward =  (self.current_empty_voxel_count / self.spc.gt_n_empty_voxels)#/self.n_images
        #self.es_similarity.append( self.spc.gt_compare_empty_voxels())
        self.solid_similarities.append(self.spc.gt_compare_solid())

        reward =  self.solid_similarities[-1] -  self.solid_similarities[-2]
        #reward =   self.es_similarity[-1] - self.es_similarity[-2]

        '''p_list = [0,50,34,2,49,3,11,14,15,150]
        if self.current_theta == p_list[self.num_steps]:
            reward = .111111111
        else:
            reward = 0.0'''

        '''dist = np.abs(self.current_theta-42)
        if dist == 0 and self.gano==False:
            reward = 5
            self.gano=True
        else:
            reward = self.minMaxNorm(dist, 0, 179 , 1 , 0)'''
        
        #if self.num_steps > 4:
        #    reward *= self.num_steps
        #reward = 0
        '''
        if self.gt_mode is True:
            #calculate increment of solid voxels ratios between gt and current volume
            gt_ratio = self.spc.gt_compare_solid()
            delta_gt_ratio = gt_ratio - self.last_gt_ratio
            self.last_gt_ratio = gt_ratio
            reward = delta_gt_ratio
        
        else:
            #get number of -1's (empty space), 0's (undetermined) and 1's (solid) from 3d volume
            self.voxel_count = [np.count_nonzero(vol == -1),
                                np.count_nonzero(vol == 0),
                                np.count_nonzero(vol == 1) ] 
            #np.histogram(self.spc.sc.values(), bins=3)[0]

           # do some calculation with the voxel count
            #calculate increment of detected spaces since last carving
            #delta = self.h[0] - self.last_vspaces_count
            #reward = min(delta,30000) / 30000
            reward=0
            self.last_voxel_count = self.voxel_count.copy() 
        '''

        if self.num_steps >= (self.n_images-1):
            reward +=  self.solid_similarities[0]
            #reward = self.spc.gt_compare_solid()
            #reward =  self.es_similarity[0]
            #reward =  self.spc.gt_compare_empty_voxels()
            #reward = self.es_similarity[-1] # -  np.mean(self.es_similarity) # self.es_similarity[-1] # - self.es_similarity[6]
            self.done = True
           
        self.total_reward += reward


        '''self.current_state = (self.im3,
                              vol.astype('float16') ,
                              np.array([self.current_theta],dtype=int),
                              np.array([self.current_phi],dtype=int))'''

        #self.current_state = ( vol.astype('float16'), np.array([self.current_theta, self.current_phi],dtype=int))

        #self.current_state = ( vol.astype('float16'), np.array(theta_state+phi_state,dtype=int))
        if self.multi_in:
            self.current_state = (self.im3, np.array([self.current_theta, self.current_phi],dtype=int))
            #self.current_state = (self.im3, np.array(theta_state+phi_state,dtype=int))
            #self.current_state = ( vol.astype('float16'), np.array([self.current_theta, self.current_phi],dtype=int))
        else:
            self.current_state = self.im3 #self.zeros #self.im3
            #self.current_state =  np.array([self.current_theta, self.current_phi],dtype=int)
            #self.current_state =  np.array([0,0],dtype=int)
            #self.current_state =  np.array(theta_state+phi_state,dtype=int)

        return self.current_state, reward, self.done, {}

 
    def minMaxNorm(self,old, oldmin, oldmax , newmin , newmax):
        return ( (old-oldmin)*(newmax-newmin)/(oldmax-oldmin) ) + newmin


    def theta_from_continuous(self, cont):
        '''converts float from (-1,1) to int (-45,45) '''
        return int(self.minMaxNorm(cont,-1.0,+1.0,-45,45))

    def phi_from_continuous(self, cont):
        '''converts float from (-1,1) to int (-3,3) '''
        return int(self.minMaxNorm(cont,-1.0,+1.0,-3.0,3.0))

   
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
