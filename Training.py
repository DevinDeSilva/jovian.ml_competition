# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:49:22 2020

@author: pc
"""

import os
import numpy as np
import seaborn as sea
import tensorflow as tf 
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.keras.applications as Applications
import tensorflow.keras.preprocessing.image as ImagePreprocessing
import cv2
import json


sea.set()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#the dataset is already nzipped
#%%
data_dir = 'F:\\Deeplearning\\Pytorch\\Competition\\jovian-pytorch-z2g\\Human protein atlas'

#loading the labels initially 
raw_labels = pd.read_csv(data_dir + '\\train.csv') 

#list of train and test images (there is no validation)
train_IDs = os.listdir(data_dir+'\\train_images')
test_IDs = os.listdir(data_dir+'\\test_images')
num_train_images = len(train_IDs)
num_test_images = len(test_IDs)
num_name_label={0: 'Mitochondria',
                1: 'Nuclear bodies',
                2: 'Nucleoli',
                3: 'Golgi apparatus',
                4: 'Nucleoplasm',
                5: 'Nucleoli fibrillar center',
                6: 'Cytosol',
                7: 'Plasma membrane',
                8: 'Centrosome',
                9: 'Nuclear speckles'}


try :
    with open('F:\\Deeplearning\\Pytorch\\Competition\\stats.txt',"r") as f:
        stats = json.load(f)
except:
    print('error')
    
try :
    with open('F:\\Deeplearning\\Pytorch\\Competition\\class_dic.txt',"r") as f:
        class_dic = json.load(f)
except:
    print('error')




