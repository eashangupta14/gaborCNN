#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this is to process images
import cv2 
import numpy as np
from utilities import math_features
import time
from numba import jit, njit, vectorize, cuda,uint32, f8, uint8
from multiprocessing import Pool

#@jit
def read(img_path):
	image = cv2.imread(img_path)
	return image

#@jit
def get_filter(ksize = 60 , sigma = 8 , theta = 0,lambd = 10 , gamma = 1, psi = np.pi *0.5):
	kernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lambd, gamma, psi)
	
	return kernel

#@jit
def get_featuremap(channels, img,theta_arr ,lambd_arr,s = 20, ksize = 60 , sigma = 8 ,  gamma = 1, psi = np.pi *0.5):
	
	a = np.empty((len(theta_arr)*s,len(lambd_arr)*s,3))
	for ch in range(0,3):
		for th in range(0,len(theta_arr)):
			for la in range(0,len(lambd_arr)):
				kernel = get_filter(theta = theta_arr[th], lambd = lambd_arr[la])
				filtered_img = cv2.filter2D(channels[ch], cv2.CV_8UC3, kernel)
				reduce_feature(filtered_img,s,a,th*s,la*s,ch)
				
	return a 
	

#@jit
def reduce_feature(fimg,s,a,x_init,y_init,z_init = -1):
	#fimg = cv2.UMat.get(fimg)
	rows = len(fimg)
	cols = len(fimg[0])
	fimg = np.multiply(fimg,fimg)
	rows_k = int(rows/20)
	cols_k = int(cols/20)
	math_features.conv(fimg,rows_k,cols_k,a,x_init,y_init,z_init,strides = s)
	
