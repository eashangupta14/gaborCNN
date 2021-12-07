#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from numba import jit, njit, vectorize, cuda,uint32, f8, uint8

#@jit
def conv(img, x_kern,y_kern,a,x_init,y_init,z_init, padding=0, strides=1):
	blur = cv2.blur(img,(x_kern,y_kern))
	for i in range(0,strides):
		for j in range(0,strides):
			x = int(x_kern/2) + (x_kern*i)
			y = int(y_kern/2) + (y_kern*j)
			a[i + x_init,j + y_init,z_init] = blur[x,y]


	
