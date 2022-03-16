
import os
import sys
import logging
import numpy as np
#import cv2
import time
import scipy.io
import glob
from keras import backend as K
import tensorflow as tf
import keras
import h5py
import unet # reso full

import numpy as np
import h5py

f = h5py.File('tr03-0005.mat','r')
#data = f.get('data/variable1')
print(f)
data = np.array(data)


#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9 
#set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def import_arousals(file_name): # target
#    import numpy as np
    f = h5py.File(file_name, 'r')
    arousals = np.transpose(np.array(f['data']['arousals']))
    return arousals

def import_signals(file_name): # feature
    return scipy.io.loadmat(file_name)['val']

def anchor (ref, ori): # input m*n np array
    d0=ori.shape[0]
    s1=float(ref.shape[1]) # size in
    s2=float(ori.shape[1]) # size out
    ori_new=ori.copy()
    for i in range(d0):
        tmp=np.interp(np.arange(s2)/(s2-1)*(s1-1), np.arange(s1), ref[i,:]) 
        ori_new[i,np.argsort(ori[i,:])]=tmp
    return ori_new

def pool_avg_2(input,if_mask=False):
    index1=np.arange(0,input.shape[1],2)
    index2=np.arange(1,input.shape[1],2)
    if (len(index2)<len(index1)):
        index2=np.concatenate((index2,[input.shape[1]-1]))
    output = (input[:,index1] + input[:,index2]) / float(2)
    if (if_mask): # -1 position are masked by -1, not avg
        mask = np.minimum(input[:,index1],input[:,index2])
        output[mask<0]=-1
    return output


image_raw = import_signals(path1 + the_id + '/' + the_id + '.mat')
d0=image_raw.shape[0]
d1=image_raw.shape[1]
image_raw = anchor(ref555, image_raw)