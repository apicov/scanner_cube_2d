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

        self.multi_in = False

        '''the state returned by this environment consiste of
        last three images,
        the volume being carved , theta position , phi position'''
        # images
        #self.images_shape = (84,84,3)
        #self.img_obs_space = gym.spaces.Box(low=0, high=255, shape=self.images_shape, dtype=np.uint8)
        
        # volume used in the carving
        self.volume_shape = (64,64,64) #(128,128,128)#(64,64,64) np.float16
        self.vol_obs_space = gym.spaces.Box(low=-1, high=1, shape=self.volume_shape, dtype=np.float16)

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
            self.observation_space = gym.spaces.Tuple((self.vol_obs_space,self.vec_ob_space))
        else:
            self.observation_space = self.vol_obs_space
            #self.observation_space = self.vec_ob_space

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
                            18:(38,0),19:(40,0),20:(42,0),21:(44,0),22:(46,0),23:(48,0),
                            24:(1,0)}'''


            '''self.actions = {0:(5,0),1:(10,0),2:(15,0),3:(20,0),4:(25,0),5:(30,0),
                            6:(35,0),7:(40,0),8:(45,0)}'''
   

            '''self.actions = {0:(2,0),1:(4,0),2:(6,0),3:(8,0),4:(10,0),5:(12,0),
                            6:(14,0),7:(16,0),8:(18,0),9:(20,0),10:(22,0),11:(24,0),
                            12:(26,0),13:(28,0),14:(30,0),15:(32,0),16:(34,0),17:(36,0),
                            18:(38,0),19:(40,0),20:(42,0),21:(44,0),22:(46,0),23:(48,0),
                            24:(50,0),25:(52,0),26:(54,0),27:(56,0),28:(58,0),29:(60,0),
                            30:(1,0)}'''


            '''self.actions = {0:(2,0),1:(4,0),2:(6,0),3:(8,0),4:(10,0),5:(12,0),
                            6:(14,0),7:(16,0),8:(18,0),9:(20,0),10:(22,0),11:(24,0),
                            12:(26,0),13:(28,0),14:(30,0),15:(32,0),16:(34,0),17:(36,0),
                            18:(1,0),
                            19:(-2,0),20:(-4,0),21:(-6,0),22:(-8,0),23:(-10,0),24:(-12,0),
                            25:(-14,0),26:(-16,0),27:(-18,0),28:(-20,0),29:(-22,0),30:(-24,0),
                            31:(-26,0),32:(-28,0),33:(-30,0),34:(-32,0),35:(-34,0),36:(-36,0),
                            37:(-1,0)}'''

            '''self.actions = {0:(0,0),1:(1,0),2:(2,0),3:(3,0),4:(6,0),5:(11,0),
                            6:(22,0),7:(45,0),8:(90,0),9:(-90,0),10:(-45,0),11:(-22,0),
                            12:(-11,0),13:(-6,0),14:(-3,0),15:(-2,0),16:(-1,0)}'''
            
            '''self.actions = {0: (0, 0),
                            1: (0, 3),
                            2: (2, 0),
                            3: (2, 3),
                            4: (6, 0),
                            5: (6, 3),
                            6: (11, 0),
                            7: (11, 3),
                            8: (22, 0),
                            9: (22, 3),
                            10: (45, 0),
                            11: (45, 3),
                            12: (-45, 0),
                            13: (-45, 3),
                            14: (-22, 0),
                            15: (-22, 3),
                            16: (-11, 0),
                            17: (-11, 3),
                            18: (-6, 0),
                            19: (-6, 3),
                            20: (-3, 0),
                            21: (-3, 3),
                            22: (-2, 0),
                            23: (-2, 3),
                            24: (-1, 0),
                            25: (-1, 3)}'''

            
            '''self.actions = {0 : (0,0), 1 : (0,1), 2 : (0,2), 3 : (0,3), 4 : (1,0), 
                            5 : (1,1), 6 : (1,2), 7 : (1,3), 8 : (2,0), 9 : (2,1), 
                            10 : (2,2), 11 : (2,3), 12 : (3,0), 13 : (3,1), 14 : (3,2), 
                            15 : (3,3), 16 : (6,0), 17 : (6,1), 18 : (6,2), 19 : (6,3), 
                            20 : (11,0), 21 : (11,1), 22 : (11,2), 23 : (11,3), 24 : (22,0), 
                            25 : (22,1), 26 : (22,2), 27 : (22,3), 28 : (45,0), 29 : (45,1), 
                            30 : (45,2), 31 : (45,3), 32 : (-45,0), 33 : (-45,1), 34 : (-45,2), 
                            35 : (-45,3), 36 : (-22,0), 37 : (-22,1), 38 : (-22,2), 39 : (-22,3), 
                            40 : (-11,0), 41 : (-11,1), 42 : (-11,2), 43 : (-11,3), 44 : (-6,0), 
                            45 : (-6,1), 46 : (-6,2), 47 : (-6,3), 48 : (-3,0), 49 : (-3,1), 
                            50 : (-3,2), 51 : (-3,3), 52 : (-2,0), 53 : (-2,1), 54 : (-2,2), 
                            55 : (-2,3), 56 : (-1,0), 57 : (-1,1), 58 : (-1,2), 59 : (-1,3), 
                            }'''


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
             
            '''self.actions = {0 : (0,0), 1 : (0,1), 2 : (0,2), 3 : (0,3), 4 : (0,-3), 
                            5 : (0,-2), 6 : (0,-1), 7 : (1,0), 8 : (1,1), 9 : (1,2), 
                            10 : (1,3), 11 : (1,-3), 12 : (1,-2), 13 : (1,-1), 14 : (2,0), 
                            15 : (2,1), 16 : (2,2), 17 : (2,3), 18 : (2,-3), 19 : (2,-2), 
                            20 : (2,-1), 21 : (3,0), 22 : (3,1), 23 : (3,2), 24 : (3,3), 
                            25 : (3,-3), 26 : (3,-2), 27 : (3,-1), 28 : (6,0), 29 : (6,1), 
                            30 : (6,2), 31 : (6,3), 32 : (6,-3), 33 : (6,-2), 34 : (6,-1), 
                            35 : (11,0), 36 : (11,1), 37 : (11,2), 38 : (11,3), 39 : (11,-3), 
                            40 : (11,-2), 41 : (11,-1), 42 : (22,0), 43 : (22,1), 44 : (22,2), 
                            45 : (22,3), 46 : (22,-3), 47 : (22,-2), 48 : (22,-1), 49 : (45,0), 
                            50 : (45,1), 51 : (45,2), 52 : (45,3), 53 : (45,-3), 54 : (45,-2), 
                            55 : (45,-1), 56 : (-45,0), 57 : (-45,1), 58 : (-45,2), 59 : (-45,3), 
                            60 : (-45,-3), 61 : (-45,-2), 62 : (-45,-1), 63 : (-22,0), 64 : (-22,1), 
                            65 : (-22,2), 66 : (-22,3), 67 : (-22,-3), 68 : (-22,-2), 69 : (-22,-1), 
                            70 : (-11,0), 71 : (-11,1), 72 : (-11,2), 73 : (-11,3), 74 : (-11,-3), 
                            75 : (-11,-2), 76 : (-11,-1), 77 : (-6,0), 78 : (-6,1), 79 : (-6,2), 
                            80 : (-6,3), 81 : (-6,-3), 82 : (-6,-2), 83 : (-6,-1), 84 : (-3,0), 
                            85 : (-3,1), 86 : (-3,2), 87 : (-3,3), 88 : (-3,-3), 89 : (-3,-2), 
                            90 : (-3,-1), 91 : (-2,0), 92 : (-2,1), 93 : (-2,2), 94 : (-2,3), 
                            95 : (-2,-3), 96 : (-2,-2), 97 : (-2,-1), 98 : (-1,0), 99 : (-1,1), 
                            100 : (-1,2), 101 : (-1,3), 102 : (-1,-3), 103 : (-1,-2), 104 : (-1,-1),
                            }'''


            '''self.actions = {0 : (-45,-3), 1 : (-45,-2), 2 : (-45,-1), 3 : (-45,0), 4 : (-45,1), 
                            5 : (-45,2), 6 : (-45,3), 7 : (-40,-3), 8 : (-40,-2), 9 : (-40,-1), 
                            10 : (-40,0), 11 : (-40,1), 12 : (-40,2), 13 : (-40,3), 14 : (-35,-3), 
                            15 : (-35,-2), 16 : (-35,-1), 17 : (-35,0), 18 : (-35,1), 19 : (-35,2), 
                            20 : (-35,3), 21 : (-30,-3), 22 : (-30,-2), 23 : (-30,-1), 24 : (-30,0), 
                            25 : (-30,1), 26 : (-30,2), 27 : (-30,3), 28 : (-25,-3), 29 : (-25,-2), 
                            30 : (-25,-1), 31 : (-25,0), 32 : (-25,1), 33 : (-25,2), 34 : (-25,3), 
                            35 : (-20,-3), 36 : (-20,-2), 37 : (-20,-1), 38 : (-20,0), 39 : (-20,1), 
                            40 : (-20,2), 41 : (-20,3), 42 : (-15,-3), 43 : (-15,-2), 44 : (-15,-1), 
                            45 : (-15,0), 46 : (-15,1), 47 : (-15,2), 48 : (-15,3), 49 : (-10,-3), 
                            50 : (-10,-2), 51 : (-10,-1), 52 : (-10,0), 53 : (-10,1), 54 : (-10,2), 
                            55 : (-10,3), 56 : (-5,-3), 57 : (-5,-2), 58 : (-5,-1), 59 : (-5,0), 
                            60 : (-5,1), 61 : (-5,2), 62 : (-5,3), 63 : (-2,-3), 64 : (-2,-2), 
                            65 : (-2,-1), 66 : (-2,0), 67 : (-2,1), 68 : (-2,2), 69 : (-2,3), 
                            70 : (0,-3), 71 : (0,-2), 72 : (0,-1), 73 : (0,0), 74 : (0,1), 
                            75 : (0,2), 76 : (0,3), 77 : (2,-3), 78 : (2,-2), 79 : (2,-1), 
                            80 : (2,0), 81 : (2,1), 82 : (2,2), 83 : (2,3), 84 : (5,-3), 
                            85 : (5,-2), 86 : (5,-1), 87 : (5,0), 88 : (5,1), 89 : (5,2), 
                            90 : (5,3), 91 : (10,-3), 92 : (10,-2), 93 : (10,-1), 94 : (10,0), 
                            95 : (10,1), 96 : (10,2), 97 : (10,3), 98 : (15,-3), 99 : (15,-2), 
                            100 : (15,-1), 101 : (15,0), 102 : (15,1), 103 : (15,2), 104 : (15,3), 
                            105 : (20,-3), 106 : (20,-2), 107 : (20,-1), 108 : (20,0), 109 : (20,1), 
                            110 : (20,2), 111 : (20,3), 112 : (25,-3), 113 : (25,-2), 114 : (25,-1), 
                            115 : (25,0), 116 : (25,1), 117 : (25,2), 118 : (25,3), 119 : (30,-3), 
                            120 : (30,-2), 121 : (30,-1), 122 : (30,0), 123 : (30,1), 124 : (30,2), 
                            125 : (30,3), 126 : (35,-3), 127 : (35,-2), 128 : (35,-1), 129 : (35,0), 
                            130 : (35,1), 131 : (35,2), 132 : (35,3), 133 : (40,-3), 134 : (40,-2), 
                            135 : (40,-1), 136 : (40,0), 137 : (40,1), 138 : (40,2), 139 : (40,3), 
                            140 : (45,-3), 141 : (45,-2), 142 : (45,-1), 143 : (45,0), 144 : (45,1), 
                            145 : (45,2), 146 : (45,3) }'''

            self.action_space = gym.spaces.Discrete(len(self.actions))

        self.zeros = np.ones((64,64,64))
        #self._spec.id = "Romi-v0"
        self.reset()

    def reset(self,theta_init=-1,phi_init=-1,theta_bias=0):
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
        #im = np.array(self.spc.get_image(self.current_theta, self.current_phi))
        # need 3 last images, this is first taken image so copy it 3 times
        #and adjust dimensions (height,width,channel)
        #self.im3 = np.array([im,im,im]).transpose(1,2,0)

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
            self.current_state = ( vol.astype('float16'), np.array([self.current_theta, self.current_phi],dtype=int))
        else:
            self.current_state = vol.astype('float16')  #self.zeros #vol.astype('float16')


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
        #im = np.array(self.spc.get_image(self.current_theta, self.current_phi))
        # need 3 last images, #and adjust dimensions (height,width,channel)
        #self.im3 = np.array([self.im3[:,:,1],self.im3[:,:,2], im]).transpose(1,2,0)

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

        reward = self.solid_similarities[-1] -  self.solid_similarities[-2] #0 #  self.es_similarity[-1] - self.es_similarity[-2]

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
            #reward =  self.spc.gt_compare_empty_voxels()
            #reward = self.es_similarity[-1]# - np.mean(self.es_similarity) #self.es_similarity[-1] - self.es_similarity[6]
            self.done = True
           
        self.total_reward += reward


        '''self.current_state = (self.im3,
                              vol.astype('float16') ,
                              np.array([self.current_theta],dtype=int),
                              np.array([self.current_phi],dtype=int))'''

        #self.current_state = ( vol.astype('float16'), np.array([self.current_theta, self.current_phi],dtype=int))

        #self.current_state = ( vol.astype('float16'), np.array(theta_state+phi_state,dtype=int))
        if self.multi_in:
            self.current_state = ( vol.astype('float16'), np.array([self.current_theta, self.current_phi],dtype=int))
        else:
            self.current_state = vol.astype('float16')   #self.zeros #vol.astype('float16')
            #self.current_state =  np.array([self.current_theta, self.current_phi],dtype=int)
            #self.current_state =  np.array([0,0],dtype=int)
            #self.current_state =  np.array(theta_state+phi_state,dtype=int)

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
