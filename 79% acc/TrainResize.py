
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:18:09 2020

@author: seraj
"""

import os
import sys
import numpy as np
import cv2




path = os.path.dirname(os.path.realpath(__file__))+"/TrainImg/"
listpicpath = os.listdir(path)
for folder in listpicpath:
    piclist=os.listdir(path+folder)
    
    
    for file in piclist:
        
        SpecTrain = cv2.imread(path + "/" + folder + "/" + file)
        new_width  = 32
        new_height = 32
        SpecTrainResize=cv2.resize(SpecTrain, (new_width,new_height))
        train_images=cv2.imwrite("TrainSpecResize/"+ folder + "/" +file ,SpecTrainResize)
        
        
# print("train_images dimentions: ", train_images.ndim)
# print("train_images shape: ", train_images.shape)
# print("train_images type: ", train_images.dtype)
#print("test_images shape: ", test_images.shape)         

              

       