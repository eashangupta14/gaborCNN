import argparse
import cv2
import helper2
import numpy as np
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
from utilities import image_processor, datamaker
from functools import partial

def b(d,q):
	d.append(q)
	

def a(q,t,p,s = 20):
	d = []
	b(d,q)

	return d
	

if __name__ == "__main__":
	wer = [1 , 45,23,67,12,0,45,-3,32,100,4,81,455]

	p = Pool()
	result = p.map(partial(a,t = 20,p = 89),wer)

	print(result)
