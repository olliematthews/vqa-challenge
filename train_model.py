# -*- coding: utf-8 -*-
"""

Creates and trains a certain model. Note that functions have been written in
the answerer class such that if you want to change the model in certain ways
you can do so while retaining weights. To do this:
    Initialise the same model as before with its params
    Load the weights
    Call the model changing function with the new params
    
    
"""
from keras.utils import to_categorical

from answererdo import Answerer
import pickle

params = {
    'lstm_layers' : [90,80],
    'dense_layers' : [1000],
    'out_dimensionality' : 1000,
    'batch_size' : 1000,
    'val_split' : 0.15,
    'lstm_do' : 0.1,
    'dense_do' : 0.4,
    'lr' : 0.0005}

# Keep these different if you don't want to overwrite when saving
savefile = 'model_save.hdf5'
loadfile = 'noim_model_save.hdf5'

# Create model
model = Answerer(params, savefile)
model.load_weights(loadfile)
model.remove_im(params)

print('Loading data')

# Can uncomment this section if the model requires question type inputs

# q_types = pickle.load(open('qtypestrain.p','rb'))
# qt_advanced = pickle.load(open('qt_advancedtrain.p','rb'))

# q_types = to_categorical(q_types, 65)
# qt_advanced = to_categorical(qt_advanced, 5)
# feed_in = [[pickle.load(open('embstrain.p','rb')),pickle.load(open('imstrain.p','rb')),q_types,qt_advanced],pickle.load(open('anstrain.p','rb'))]
# feed_in = [[pickle.load(open('embstrain.p','rb')),pickle.load(open('imstrain.p','rb'))],pickle.load(open('anstrain.p','rb'))]

feed_in = [[pickle.load(open('embstrain.p','rb'))],pickle.load(open('anstrain.p','rb'))]

# Train for 5 epochs
model.learn(feed_in, 5)