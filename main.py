#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utilities import image_processor,datamaker
from model import cnn
import helper
import cv2
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from tensorflow_addons.layers import SpatialPyramidPooling2D
import numpy as np
import math
import os
from numba import jit, njit, vectorize, cuda,uint32, f8, uint8
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report
from mlxtend.plotting import plot_confusion_matrix
import matplotlib
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pandas as pd
#import torch
#import torchvision
#from torch.utils.data import Dataset, DataLoader
#import torch.nn as nn
#from torch.nn import functional as F
#from GaborNet import GaborConv2d


#@jit
def makedata():
	#theta = [0,np.pi*.25,np.pi*.5,np.pi*.75,np.pi]
	#lambd = [10,20,30,40,50,55]
	#image(theta,lambd)
	jg_x = []
	jg_y = []
	pusa_x = []
	pusa_y = []

	jg62_data, pusa372_data = datamaker.datamake()
	for features, label in jg62_data:
		jg_x.append(features)
		jg_y.append(label)

	for features, label in pusa372_data:
		pusa_x.append(features)
		pusa_y.append(label)

	np.save('features_jg.npy',jg_x)
	np.save('label_jg.npy',jg_y)
	np.save('features_pusa.npy',pusa_x)
	np.save('label_pusa.npy',pusa_y)

#@jit
def loaddata():

	'''

	x = np.load('features_jg.npy')
	y = np.load('label_jg.npy')
	x1 = np.load('features_pusa.npy')
	y1 = np.load('label_pusa.npy')

	'''
	# to load data
	x = np.load('/content/try6/features_jg.npy')
	y = np.load('/content/try6/label_jg.npy')
	x1 = np.load('/content/try6/features_pusa.npy')
	y1 = np.load('/content/try6/label_pusa.npy')
	#'''

	return x,y,x1,y1

#@jit
def main():

	#pre processing step
	#makedata()

	#'''
	# cnn using tensorflow
	
	#loading data
	x,y,x1,y1 = loaddata()

	x = x[2521:]
	y = y[2521:]

	
	#divide data
	jg_x_train, jg_x_test, jg_y_train, jg_y_test = datamaker.divide_data(x,y)
	jg_y_train = to_categorical(jg_y_train)
	jg_y_test = to_categorical(jg_y_test)

	pusa_x_train, pusa_x_test, pusa_y_train, pusa_y_test = datamaker.divide_data(x1,y1)
	pusa_train = to_categorical(pusa_y_train)
	pusa_y_test = to_categorical(pusa_y_test)
	
	
	
	our_model = cnn.models(x)
	#our_model.layers[0].trainable = False
	our_model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),optimizer = Adam(learning_rate = .0005), metrics = ['accuracy'])
	history = our_model.fit(jg_x_train,jg_y_train, batch_size = 32, epochs = 28 )
	our_model.summary()
	our_model.evaluate(jg_x_test,jg_y_test)
	
	#y_predict = our_model.predict_classes(jg_x_test)
	y_predict = our_model.predict(jg_x_test)
	#print(y_predict.shape)
	#print(y_predict())
	y_predict = np.argmax(y_predict, axis=-1) 
	#print(y_predict.shape)
	#print(y_predict())
	y2 = np.argmax(jg_y_test, axis=-1)


	fig1 = plt.figure()
	ax1 = fig1.add_subplot(2,1,1)
	ax1.plot(history.history['accuracy'])
	#plt.subplot(2,1,1)
	#plt.plot(history.history['accuracy'])
	plt.title('model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	#plt.legend(['train','test'], loc = 'upper left')

	ax2 = fig1.add_subplot(2,1,2)
	ax2.plot(history.history['loss'])
	#plt.subplot(2,1,2)
	#plt.plot(history.history['loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('Epoch')
	#plt.legend(['train','test'], loc = 'upper left')
	plt.tight_layout()
	fig1.savefig('foo.png')

	
	
	matplotlib.rcParams.update(matplotlib.rcParamsDefault)
	
	mat = confusion_matrix(y2,y_predict)
	class_names = ['before flowering','control','young seedling']
	#df_cm = pd.DataFrame( mat, index=class_names, columns=class_names)
	#fig = plt.figure(figsize=figsize)
	#heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
	#print(classification_report(y2, y_predict))

	fig, ax = plot_confusion_matrix(conf_mat = mat,figsize = (6,6), class_names = class_names, show_normed = False)
	plt.tight_layout()
	fig.savefig('cm.png')

	#'''
	

	'''
	#cnn using pytorch

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	class GaborNN(nn.Module):
		def __init__(self):
			super(GaborNN, self).__init__()
			self.pool = nn.MaxPool2d(2,2)
			self.g0 = GaborConv2d(in_channels=3, out_channels=32, kernel_size=(15, 15))
			self.bn0 = nn.BatchNorm2d(32)
			self.c1 = nn.Conv2d(32, 64, (5,5),stride = 2)
			self.bn1 = nn.BatchNorm2d(64)
			self.c2 = nn.Conv2d(64, 64, (3,3),stride = 1)
			self.bn2 = nn.BatchNorm2d(64)
			self.fc1 = nn.Linear(64*4*9, 3)
		def forward(self, inp):
			inp = self.pool(F.leaky_relu(self.g0(inp)))
			inp = self.bn0(inp)
			inp = self.pool(F.relu(self.bn1(self.c1(inp))))
			inp =  nn.Dropout(.2)(inp)
			#print(inp.shape)
			inp = self.pool(F.relu(self.bn2(self.c2(inp))))
			inp =  nn.Dropout(.2)(inp)
			print(inp.shape)
			inp = inp.view(inp.size(0), -1)
			inp =  nn.Dropout(.2)(inp)
			inp = F.softmax(self.fc1(inp))
			return inp


	x3,y3,x4,y4 = loaddata()
	x3 = x3[1681:]
	y3 = y3[1681:]
	jg_x_train, jg_x_test, jg_y_train, jg_y_test = datamaker.divide_data(x3,y3)
	jg_x_train, jg_x_validate, jg_y_train, jg_y_validate = datamaker.divide_data(jg_x_train,jg_y_train)

	class plant_train(Dataset):

		def __init__(self):
			jg_x_train_new = np.transpose( jg_x_train, (0,3,1,2))
			
			self.x_train = torch.from_numpy(jg_x_train_new)
			self.y_train = torch.from_numpy(jg_y_train)
			self.nsamples_train = jg_y_train.shape[0]

		def __getitem__(self, index):
			return self.x_train[index] , self.y_train[index]

		def __len__(self):

			return self.nsamples_train 

	class plant_test(Dataset):

		def __init__(self):
			jg_x_test_new = np.transpose( jg_x_test, (0,3,1,2))
			self.x_test = torch.from_numpy(jg_x_test_new)
			self.y_test = torch.from_numpy(jg_y_test)
			self.nsamples_test = jg_y_test.shape[0]

		def __getitem__(self, index):
			return self.x_test[index] , self.y_test[index]

		def __len__(self):

			return self.nsamples_test

	class plant_validate(Dataset):

		def __init__(self):
			jg_x_validate_new = np.transpose( jg_x_validate, (0,3,1,2))
			self.x_validate = torch.from_numpy(jg_x_validate_new)
			self.y_validate = torch.from_numpy(jg_y_validate)
			self.nsamples_validate = jg_y_validate.shape[0]

		def __getitem__(self, index):
			return self.x_validate[index] , self.y_validate[index]

		def __len__(self):

			return self.nsamples_validate


	dataset_train = plant_train()
	dataloader_train = DataLoader(dataset = dataset_train, batch_size= 32, shuffle = True)

	dataset_test = plant_test()
	dataloader_test = DataLoader(dataset = dataset_test, batch_size= 32, shuffle = False)

	dataset_validate = plant_validate()
	dataloader_validate = DataLoader(dataset = dataset_validate, batch_size= 32, shuffle = False)

	

	net = GaborNN().to(device)
	cost_function = nn.CrossEntropyLoss()
	num_epochs = 12
	total_samples_train = len(dataset_train)
	number_iteration_train = math.ceil(total_samples_train/32)
	total_samples_test = len(dataset_test)
	number_iteration_test = math.ceil(total_samples_test/32)
	optimizer = torch.optim.Adam(net.parameters(), lr=0.0001) 
	pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
	print(pytorch_total_params)

	net = net.float()
	net.train()
	for epoch in range(num_epochs):
		losses = []
		closs = 0
		for i, (feature,label) in enumerate(dataloader_train,0):

			label = label.type(torch.LongTensor)
			feature,label = feature.to(device),label.to(device)
			prediction = net(feature.float())
			#value,prediction = torch.max(prediction,1)
			loss = cost_function(input = prediction, target = label)
			
			closs += loss.item()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print('epoch' + str(epoch) + "---->" + 'loss = ' + str(closs))
		with torch.no_grad():
			correct=0
			tot=0

			for batches in dataloader_train:
			    data,output = batches
			    output = output.type(torch.LongTensor)
			    data,output = data.to(device),output.to(device)
			    prediction = net(data.float())
			    _,prediction = torch.max(prediction.data,1)  #returns max as well as its index
			    tot += output.size(0)
			    correct += (prediction==output).sum().item()
			print('Train Accuracy = '+str((correct/tot)*100))

			correct=0
			tot=0
			valloss = 0
			for batches in dataloader_validate:
			    data,output = batches
			    output = output.type(torch.LongTensor)
			    data,output = data.to(device),output.to(device)
			    prediction = net(data.float())
			    loss = cost_function(input = prediction, target = output)
			    valloss += loss.item()
			    _,prediction = torch.max(prediction.data,1)  #returns max as well as its index
			    tot += output.size(0)
			    correct += (prediction==output).sum().item()
			print('validation loss = ' + str(valloss) + ' validate Accuracy = '+str((correct/tot)*100))
		
	with torch.no_grad():
		correctHits=0
		total=0
		for batches in dataloader_test:
		    data,output = batches
		    output = output.type(torch.LongTensor)
		    data,output = data.to(device),output.to(device)
		    prediction = net(data.float())
		    _,prediction = torch.max(prediction.data,1)  #returns max as well as its index
		    print(prediction)
		    print(output)
		    total += output.size(0)
		    correctHits += (prediction==output).sum().item()
		print('Test Accuracy = '+str((correctHits/total)*100))

	'''


if __name__ == "__main__":
    
    main()







