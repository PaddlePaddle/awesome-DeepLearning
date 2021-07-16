import numpy as np

from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Reshape
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.applications.resnet50 import ResNet50,preprocess_input
#from keras.applications.alexnet import AlexNet,preprocess_input
#from keras.applications.vgg16 import VGG16,preprocess_input
from keras.models import model_from_json,model_from_config,load_model
from keras.optimizers import SGD,RMSprop,adam,Adam
from keras.preprocessing import image
from keras import backend as K
from keras import models
from keras import layers as lay
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

import cv2
from PIL import Image
from sklearn.metrics import mean_squared_error
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numpy import *
import os
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

K.set_image_data_format('channels_last')
# sets the image shape specification to be (num_images,num_rows,num_columns,num_channels) 
#as if the backend is Theano num_channels come as the second argument

def get_model():
# function to make the model and compile it

	base_model = ResNet50()
	#base_model = VGG16(weights='imagenet', include_top=True)
	# imports the base Resnet50 model and stores it in base_model along with the preloaded ImageNet weights 

	base_model.summary()
	#prints the summary of the resnet model
	
	fl1 = base_model.get_layer('predictions').output
	#get the output of the layer of resnet with the name flatten_1
	fc1=Dense(1000,activation='relu')(fl1)
	#add a fully connected layer with 1000 neurons to the fl1 layer and apply relu activation on it

	drop1=Dropout(0.3)(fc1)
	#add a dropout of 0.3 i.e kill 30% of the neurons at random to introduce generecity and avoid overfitting

	fc2=Dense(400,activation='relu')(drop1)
	drop2=Dropout(0.3)(fc2)
	# add the fully connected layer with 500 neurons and apply a dropout of 0.3 same as the previous layer
	fc3=Dense(150,activation='relu')(drop2)
	drop3=Dropout(0.3)(fc3)
	fc4=Dense(40,activation='relu')(drop3)
	drop4=Dropout(0.3)(fc4)
	#fc5=Dense(50,activation='relu')(drop4)
	#drop5=Dropout(0.3)(fc5)
	#fc6=Dense(20,activation='relu')(drop5)
	#drop6=Dropout(0.3)(fc6)
	predictions = Dense(3,activation='softmax')(drop4)
	# finally drop the model to 2 class prediction since we have to classify between two classes 
	#and apply softmax activation since it gives us the class probabilities between 0 and 1

	# till now layers have been stacked onto one another


	model = Model(base_model.input, predictions)
	# specifies a model whose input layer is the input layer of the resnet model and output layer is 
	# the predictions layer which gives the class probabilities of the two classes

	for layer in base_model.layers:
		layer.trainable=False
	# since resnet is already trained on the imagenet dataset it has already learned the basic features
	# such as lines , curves etc and thus we freeze the resnet model to avoid computation and save time
	# Hence these two lines set all the layers in the resnet model to not be trainable so that only the 
	# fully connected layers that we have added are trainable

	adam=Adam(lr=0.001)
	# define a customised adam optimizer with a learning rate of 0.001. 
	# You can also set other parameters such as momentum and decay

	model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',metrics=['accuracy'])
	# compile the model means to create a computation graph so that it knows that it has to use the
	# adam optimizer and compute the categorical cross-entropy loss during back propogation
	# also we define which all metrics we have to take track of. Loss is by default and we have
	# accuracy as well. But till now the model is just a graph and no data has been fed into it.

	model.summary()
	# prints the summary of the whole model we have just created

	return model
	# return the made and compiled model

def get_input():
	X=[]
	label=[]
	path1="no_aug"
	path2="pathological_aug"
	path3="high_aug"
	list1=os.listdir(path1)
	list2=os.listdir(path2)
	list3=os.listdir(path3)
	for elem in list1:
		img=image.load_img(path1+"/"+elem,target_size=(224,224))
		x=image.img_to_array(img)
		X.append(x)
		label.append(0)
	for elem in list2:
		img=image.load_img(path2+"/"+elem,target_size=(224,224))
		x=image.img_to_array(img)
		X.append(x)
		label.append(1)
	for elem in list3:
		img=image.load_img(path3+"/"+elem,target_size=(224,224))
		x=image.img_to_array(img)
		X.append(x)
		label.append(2)
	X=preprocess_input(np.array(X))
	label=np.array(label)
	print ("input taken")
	return X,label

def fit_model(model,X,label):
# function to train our model on the data given by X and its groundtruth given by label

	X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=2)
	# split the data as 20% testing and 80% training with a random state of 2. Random state means that
	# everytime the data is randomly shuffled and then split but using the same random state means that
	# the data is shuffled similarly every time but if you change it to any other number it will shuffle
	# differently

	model_json = model.to_json()
	# converts the model architecture into a json file

	with open("model.json", "w") as json_file:
		json_file.write(model_json)
		print("json saved")
	# writes the model architecture into the json file named as model.json

   	#checkpoint=ModelCheckpoint("Model.h5",monitor='val_acc',mode='max',save_best_only=True,save_weights_only=True)
   	# creates a checkpoint such that after every epoch it automatically saves the model weights 
   	# monitor defines what we have to keep track of which in this case is validation accuracy.For
   	# training accuracy you can write acc and for loss write loss or val_loss
   	# Model.h5 defines the path where we want to save the weights
   	# mode defines when we have to save the weights when the thing being monitored is maximum or minimum
   	# save_best_only set to True means that only the best model is saved and save_weights_only means that
   	# only the model weights are stored since we have already stored the json file

   	# *********** it is always recommended to save the json and weights independently instead of the whole model at once **********

   	#callbacks_list=[checkpoint]
   	# creates a list of the checkpoint
	

	
	

	model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=15, batch_size=32)
	#callbacks=callbacks_list
	# fits the model i.e provides the data to the built model to train it
	# X_train and y_train are the training data and the corresponding groundtruth
	# X_test and y_test are the validation data on which the trained model will be tested after every epoch
	# 15 epochs means that 15 iterations of training will be done
	# batch_size=32 refers 
	scores = model.evaluate(X_test, y_test, verbose=0)
	print ("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

X,label=get_input()
X,label=shuffle(X,label,random_state=2)
model=get_model()
fit_model(model,X,label)

