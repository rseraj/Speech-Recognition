# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:20:34 2020

@author: seraj
"""

import os
import sys
import numpy as np
import cv2





path = os.path.dirname(os.path.realpath(__file__))+"/TestImg/"
listpicpath = os.listdir(path)
for folder in listpicpath:
    piclist=os.listdir(path+folder)
    
    
    for file in piclist:
        
        SpecTest = cv2.imread(path + "/" + folder + "/" + file)
        new_width  = 32
        new_height = 32
        SpecTestResize=cv2.resize(SpecTest, (new_width,new_height))
        cv2.imwrite("TestSpecResize/"+ folder + "/" +file ,SpecTestResize)
        
        