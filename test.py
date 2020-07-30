# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 16:32:23 2020

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
import pickle





#%%

tensor  = [[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]],[[19,20,21],[22,23,24],[25,26,27]]]
tensor = np.array(tensor)
#print(tensor**2)
print(' ')
print(np.square(tensor))


print('*********************************************************')

img = cv2.imread('F:\\Deeplearning\\Pytorch\\Competition\\jovian-pytorch-z2g\\Human protein atlas\\train_images\\0.png')
print(img.shape)
img=(img).reshape(-1,3)
print(img)
img_not_aug = img
img_with_aug = np.array(np.square(img))