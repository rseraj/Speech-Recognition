# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:58:33 2020

@author: seraj
"""


import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
import sys


path = os.path.dirname(os.path.realpath(__file__))+"/TestSet/"
listpicpath=os.listdir(path)
for folder in listpicpath:
    piclist=os.listdir(path+folder)
    for file in piclist:
        sample_rate, samples = wavfile.read(path+"/"+folder+"/"+file)
        fig,ax=plt.subplots(1)
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        ax.axis('off')
        pxx,freqs,bins,im=plt.specgram(samples, Fs=sample_rate)
        fig.savefig("TestImg/"+folder+"/"+file+".png")