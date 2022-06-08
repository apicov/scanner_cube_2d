import glob
import numpy as np
import os
import time
import cv2
from PIL import Image



'''def get_mask(img):
   res=np.array(img)[:,:,:3]
   mask1=np.all((res==[57,57,57]), axis=-1) 
   mask2=np.all((res==[58,58,58]), axis=-1) 
   mask3=np.all((res==[56,56,56]), axis=-1)
   mask12=np.bitwise_or(mask1,mask2)
   mask_inv=np.bitwise_or(mask12,mask3)
   mask=np.bitwise_not(mask_inv)
   return 255*mask'''


def get_mask(img):
   # Make Numpy array
    ni =np.array(img)[:,:,:3]# remove alpha
    background = np.logical_and( ni[:,:,:]>[45,45,45],  ni[:,:,:]< [64,64,64])
    background = np.all(background,axis=-1) #make one channel
    background = np.logical_not(background)#background in black
    return (background*255).astype(np.uint8)



# Parent Directory path 
parent_dir = "/home/pico/uni/romi/scanner-gym_models_v3"
# Directory 
directory =  '213_2d'
# Path 
model_path = os.path.join(parent_dir, directory)

masks_dir  = os.path.join(model_path, 'masks')
if not os.path.exists(masks_dir):
    os.makedirs(masks_dir)


# get all .png file names of images from folder path
img_files = sorted (glob.glob(os.path.join(model_path, 'imgs', '*.png')) )

for i in img_files:
    img = Image.open(i)
    mask = get_mask(img)
    cv2.imwrite(os.path.join(masks_dir,i[-11:-4]+'.png'),mask)
