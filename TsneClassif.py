#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:08:34 2018

@author: Tristan
"""
#%%
from __future__ import print_function
import keras
#from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, GaussianDropout, GaussianNoise, Activation, Lambda

import math
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))



#%% AUGMENTATION

import os

from joblib import Parallel, delayed
from tqdm import tqdm_notebook as tqdm
import pickle
import pandas as pd

import jams
import numpy as np

import muda

def root(x):
    return os.path.splitext(os.path.basename(x))[0]

AUDIO = jams.util.find_with_extension('audio', 'wav')

print("Number of files : " + str(len(AUDIO)))

#%% DATA PUMP
#from glob import glob
import pumpp

#Change the TSNE dimension
tsnedim = 2

# Build a pump
sr = 44100
hop_length = 4096

p_feature = pumpp.feature.CQTMag(name='cqt', sr=sr, hop_length=hop_length, log=True, conv='tf', n_octaves=6)
p_chord_tag = pumpp.task.ChordTagTransformer(name='chord_tag', sr=sr, hop_length=hop_length, sparse=True)
p_chord_struct = pumpp.task.ChordTransformer(name='chord_struct', sr=sr, hop_length=hop_length, sparse=True)

pump = pumpp.Pump(p_feature, p_chord_tag, p_chord_struct)

# Save the pump
with open('pump.pkl', 'wb') as fd:
    pickle.dump(pump, fd)
    
# Transformation with CQT
def convert(aud, pump, outdir):
    data = pump.transform(aud)
    fname = os.path.extsep.join([root(aud), 'npz'])
    np.savez(os.path.join(outdir, fname), **data)

    
OUTDIR = 'pump/'
#%%
if not os.path.exists(OUTDIR):
 os.makedirs(OUTDIR)

if os.listdir(OUTDIR)==[]:
	Parallel(n_jobs=1, verbose=10)(delayed(convert)(aud, pump, OUTDIR) for aud in AUDIO);

transformOptions = {}    
transformOptions["hopSize"] = 4096    
transformOptions["contextWindows"] = 15
transformOptions["resampleTo"] = 44100

# Create the dataset
class audioSet:
    def __init__(self):
        self.data = []
        self.metadata = {}
        self.metadata['chord'] = []
audioSetTsne = audioSet()

for rootf, dirs, filenames in os.walk(OUTDIR):
    print("loading data")
    
for i in range(len(filenames)):
    data = np.load(rootf+filenames[i])
    #data = np.load(fname)
    d2 = dict(data)
    data.close()
    data = d2
    data['cqt/mag'] = data['cqt/mag'][0]
    audioSetTsne.data.append(data['cqt/mag'])
    
for k in range(len(filenames)):
    maxFrame = len(audioSetTsne.data[k])
    finalData = {}
    for numFrame in range(maxFrame - transformOptions["contextWindows"]  + 1):
        finalData[numFrame] = audioSetTsne.data[k][range(numFrame, numFrame + transformOptions["contextWindows"])];
    audioSetTsne.data[k] = finalData
#%%
from keras.models import load_model
from numpy import linalg as LA
from sklearn.manifold import TSNE

#%%
model = load_model("model_ok.hdf5")
model.summary()
#%%

try:
	X_embedded = np.loadtxt('tsne' + str(tsnedim) + '.txt')
except IOError:
	output = {}    
	for k in range(len(filenames)):
		dictlist = []
		for key, value in audioSetTsne.data[k].items():
			#temp = [key,value]
			dictlist.append(value)
		inputs = np.asarray(dictlist)
		output[k] = model.predict({"inputs" : inputs})
		print("Done " + str(k) + " over " + str(len(filenames)))
	#%%
	X = []
	for k in range(len(filenames)):
		out = output[k][1]
		a = np.mean(out, axis=0)
		b = LA.norm(out, axis=0)
		c = np.ndarray.max(out, axis=0)
		d= np.asarray([a,b,c]).reshape(600)
		X.append(d)
	
	if tsnedim == 3:
		X_embedded = TSNE(n_components=3).fit_transform(X)
	else:
		X_embedded = TSNE(n_components=2).fit_transform(X)
	np.savetxt('tsne' + str(tsnedim) + '.txt', X_embedded)	
	X_embedded.shape

	

for i in range(len(filenames)):
	print(str(i) + " : " + filenames[i])
	
#%%
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from pylab import figure

if tsnedim == 3:
	fig = figure()
	ax = Axes3D(fig)
	#ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2], filenames)    
	for i in range(len(filenames)): #plot each point + it's index as text above
		ax.scatter(X_embedded[i,0],X_embedded[i,1], X_embedded[i,2], color='b') 
		ax.text(X_embedded[i,0],X_embedded[i,1], X_embedded[i,2], '%s' % (i), size=12, zorder=1, color='k')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	pyplot.show()

else:
	fig = figure()
	ax = fig.add_subplot(111)
	for i in range(len(filenames)): #plot each point + it's index as text above
		ax.scatter(X_embedded[i,0],X_embedded[i,1], color='b') 
		ax.text(X_embedded[i,0],X_embedded[i,1], '%s' % (i), size=12, zorder=1, color='k')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	pyplot.show()
	#plt.scatter(X_embedded[:,0], X_embedded[:,1], color='b')
	#plt.show()

