#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, MaxPooling2D, BatchNormalization, GRU, RandomFlip
from tensorflow_addons.layers import SpatialPyramidPooling2D
import numpy as np
from numba import jit, njit, vectorize, cuda,uint32, f8, uint8
import cv2
from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_data_format('channels_last')

from tensorflow.keras import layers	


def custom_gabor(shape, dtype=None):
    pi = np.pi
    orientation_spread = np.array([0, pi/4, pi/2, pi*3/4, pi])
    scales = np.array([10,20,30,40,50,55])
    real_kernels = []
    img_kernels = []
#     size, sigma, theta, lambda, gamma aspect ratio
    for orientation in orientation_spread:
        for scale in scales:
            real_kernel = cv2.getGaborKernel((5, 5), 1, orientation, scale, 1, 0)
            imaginary_kernel = cv2.getGaborKernel((5, 5), 1, orientation, scale, 1, np.pi / 2)
            real_kernels.append(real_kernel)
            img_kernels.append(imaginary_kernel)
    stacked_list = np.array(real_kernels)
    stacked_list = np.array([stacked_list, stacked_list, stacked_list])
    #print(stacked_list.shape)
    # stack number equal to number of color channel RGB: ([stacked_list, stacked_list, stacked_list])
    #stacked_list = np.array([stacked_list])
    stacked_list = np.einsum('hijk->jkhi', stacked_list)
    #print(stacked_list.shape)

    stacked_list = K.variable(stacked_list)
    random = K.random_normal(shape, dtype=dtype)
    return stacked_list

#@jit(target = cuda)
def models(x):
	model = Sequential()
	#model.add(BatchNormalization())
	#model.add(RandomFlip(mode = "horizontal",input_shape=x.shape[1:]))
	model.add(Conv2D(30, (5, 5), padding='same',kernel_initializer=custom_gabor, data_format='channels_last', input_shape=x.shape[1:]))
	#model.add(Conv2D(30, (5, 5), padding='same',kernel_initializer=custom_gabor, data_format='channels_last'))
	model.add(BatchNormalization())
	#model.add(Activation('relu'))
	#model.add(MaxPooling2D(pool_size=(4, 4)))
	model.add(Conv2D(64, (5,5), strides = 2))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3,3), strides = 1))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	#model.add(Dropout(.2))
	model.add(MaxPooling2D(pool_size=(4, 4)))
	#model.add(Conv2D(128, (3,3), strides = 1))
	#model.add(BatchNormalization())
	#model.add(Activation('relu'))
	#model.add(Dropout(.2))
	#model.add(Conv2D(128, (3,3), strides = 1))
	#model.add(BatchNormalization())
	#model.add(Activation('relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(SpatialPyramidPooling2D([1,4,8]))
	#model.add(GRU(1))
	model.add(Flatten())
	model.add(Dropout(.5))
	#model.add(Dense(1024,activation ='softmax' ))
	model.add(Dense(3,activation ='softmax'))
	#model.add(Activation('softmax'))
	return model

