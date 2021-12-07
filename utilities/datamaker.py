#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from utilities import image_processor
from sklearn.model_selection import train_test_split
import time
from numba import jit, njit, vectorize, cuda,uint32, f8, uint8
import pywt

datadir = "C:/Users/eashan/Desktop/agriculture/data" 
datanum = ["data10","data20","data30","data40"]
categories = ["Before flowering", "Control", "Young Seedling"]

#jit
def image(img,theta, lambd):
	#x = int(img.shape[0]/2)
	#y = int(img.shape[1]/2)
	#img = img[150:x+1200,y-600:y+900,:]
	channels = [img[:,:,0],img[:,:,1],img[:,:,2]]   #channels of the image
	feature_map = image_processor.get_featuremap(channels,img,theta,lambd)  # getting the featuremap
	return feature_map

#@jit
def datamake():
	data_array_jg62 = []
	data_array_pusa372 = []
	for num in datanum:
		path = os.path.join(datadir,num)
		for date in os.listdir(path):
			s1 = time.time()
			path2 = os.path.join(path,date)
			for category in os.listdir(path2):
				class_num = os.listdir(path2).index(category)
				path3 = os.path.join(path2,category)
				for plant_type in os.listdir(path3):
					path4 = os.path.join(path3,plant_type)
					for potnum in os.listdir(path4):
						path5 = os.path.join(path4,potnum)
						for img in os.listdir(path5):

							img_array = cv2.imread(os.path.join(path5,img))
							#image resize
							img_array = img_array[500:3000,:,:]
							img_array = cv2.resize(img_array, (0,0), fx=0.15, fy=0.2)
							img_array = img_array[:450,:,:]
							#img_array = cv2.resize(img_array, (0,0), fx=0.2, fy=0.3)

							abc = pywt.dwt2(img_array[:,:,0],'haar', mode = 'periodization')
							cA11, (cH,cV,cD) = abc
							
							abc = pywt.dwt2(cA11,'haar', mode = 'periodization')
							cA, (cH,cV,cD) = abc

							cA = cA + np.amin(cA)
							cA = cA/np.amax(cA)
							cA = cA*255

							

							abc2 = pywt.dwt2(img_array[:,:,1],'haar', mode = 'periodization')
							cA2, (cH,cV,cD2) = abc2
							
							abc2 = pywt.dwt2(cA2,'haar', mode = 'periodization')
							cA2, (cH,cV,cD) = abc2

							cA2 = cA2 + np.amin(cA2)
							cA2 = cA2/np.amax(cA2)
							cA2 = cA2*255



							abc3 = pywt.dwt2(img_array[:,:,2],'haar', mode = 'periodization')
							cA3, (cH,cV,cD3) = abc3
							
							abc3 = pywt.dwt2(cA3,'haar', mode = 'periodization')
							cA3, (cH,cV,cD) = abc3

							cA3 = cA3 + np.amin(cA3)
							cA3 = cA3/np.amax(cA3)
							cA3 = cA3*255

							h,w = cA.shape


							a = np.empty((h,w,3),dtype=int)
							a[:,:,0] = cA3
							a[:,:,1] = cA2
							a[:,:,2] = cA
							a = a.astype(np.uint8)

							img_array = a
							

							#theta = [0,np.pi*.25,np.pi*.5,np.pi*.75,np.pi]  #different theta value for gabor filter
							#lambd = [10,20,30,40,50,55]   #different labmda value for gabor filter
							
							#img_array = image(img_array,theta,lambd)   # callng the image function which returns the feature map

							#adding feature maps to the array.
							if plant_type == "JG-62":
								data_array_jg62.append([img_array,class_num])
							else:
								data_array_pusa372.append([img_array,class_num])


			print(time.time()-s1)
			print(date + "done")



	return data_array_jg62,data_array_pusa372

#@jit
def divide_data(feature, labels):
	X_train, X_test, y_train, y_test = train_test_split(feature, labels, test_size=0.2, random_state=240)
	return X_train, X_test, y_train, y_test
