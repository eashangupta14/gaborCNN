import argparse
import cv2
import helper2
import numpy as np
import os
#import main
import matplotlib.pyplot as plt
from multiprocessing import Pool
from utilities import image_processor, datamaker
from functools import partial
import time
import tensorflow as tf 
from tensorflow.keras.utils import to_categorical
import pywt
from PIL import Image



if __name__ == "__main__":
	
	'''
	x,y,x1,y1 = main.loaddata()

	
	jg_y_train = to_categorical(y)
	y2 = np.argmax(jg_y_train, axis=-1)
	print((y2 == y).all())
	'''


	
	'''
	y = 100
	x = 450
	img = cv2.imread('C:/Users/eashan/Desktop/agriculture/data/for_try/IMG_4089.jpg')
	
	img3 = img[500:3000,:,:]
	small = cv2.resize(img3, (0,0), fx=0.15, fy=0.2) 
	small = small[:450,:,:]
	#small = cv2.resize(small, (0,0), fx=0.2, fy=0.3)
	print("shape = ")
	print(small.shape)
	#small = small[y:x,:,:]

	abc = pywt.dwt2(small[:,:,0],'haar', mode = 'periodization')
	cA11, (cH,cV,cD) = abc
	
	abc = pywt.dwt2(cA11,'haar', mode = 'periodization')
	cA, (cH,cV,cD) = abc

	cA = cA + np.amin(cA)
	cA = cA/np.amax(cA)
	cA = cA*255

	print(cA.shape)

	abc2 = pywt.dwt2(small[:,:,1],'haar', mode = 'periodization')
	cA2, (cH,cV,cD2) = abc2
	
	abc2 = pywt.dwt2(cA2,'haar', mode = 'periodization')
	cA2, (cH,cV,cD) = abc2


	print(cA.shape)

	abc3 = pywt.dwt2(small[:,:,2],'haar', mode = 'periodization')
	cA3, (cH,cV,cD3) = abc3
	
	abc3 = pywt.dwt2(cA3,'haar', mode = 'periodization')
	cA3, (cH,cV,cD) = abc3

	cA2 = cA2 + np.amin(cA2)
	cA2 = cA2/np.amax(cA2)
	cA2 = cA2*255

	cA3 = cA3 + np.amin(cA3)
	cA3 = cA3/np.amax(cA3)
	cA3 = cA3*255

	h,w = cD.shape


	a = np.empty((h,w,3),dtype=int)
	a[:,:,0] = cA3
	a[:,:,1] = cA2
	a[:,:,2] = cA


	a = a.astype(np.uint8)
	'''
	'''
	img2 = cv2.imread('C:/Users/eashan/Desktop/agriculture/data/for_try/IMG_6782.jpg')
	img2 = img2[1000:3550,:,:]
	small2 = cv2.resize(img2, (0,0), fx=0.15, fy=0.2) 
	#small2 = small2[y:x,:,:]
	print(cA)

	cA = cA.astype(np.uint8)
	print(cA)

	'''
	#plt.imshow(img)
	#plt.show()


	#plt.imshow(img2)
	#plt.show()
	'''

	plt.figure()
	plt.subplot(2,2,1)
	plt.imshow(img)

	plt.subplot(2,2,2)
	plt.imshow(small)

	plt.subplot(2,2,3)
	plt.imshow(a)

	plt.subplot(2,2,4)
	plt.imshow(small2)
	plt.show()
	'''

	
	img_array = cv2.imread('C:/Users/eashan/Desktop/agriculture/data/for_try/IMG_4089.jpg')
	wer = img_array
	wer2 = np.empty(wer.shape,dtype=int)
	wer2[:,:,0] = wer[:,:,2]
	wer2[:,:,1] = wer[:,:,1]
	wer2[:,:,2] = wer[:,:,0]
	print(wer.shape)
	#image resize
	img_array = img_array[500:3000,:,:]
	img_array = cv2.resize(img_array, (0,0), fx=0.15, fy=0.2)
	img_array = img_array[:450,:,:]
	print(img_array.shape)
	img_array1 = cv2.resize(img_array, (0,0), fx=0.2, fy=0.3)
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
	print(a.shape)
	img_array = a

	plt.figure()
	plt.subplot(1,2,1)
	plt.imshow(wer2)

	plt.subplot(1,2,2)
	plt.imshow(img_array)
	plt.show()
	plt.show()

	print(1)