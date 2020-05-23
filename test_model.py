# -*- coding: utf-8 -*-
"""
The code used to test the model, and provide the results file in the required 
form. 
"""
from keras.utils import to_categorical

import json
import spacy
import numpy as np
from load_features import load_json
from answererdo import Answerer
import pickle


params = {
    'lstm_layers' : [90,80],
    'dense_layers' : [1000],
    'out_dimensionality' : 1000,
    'batch_size' : 1000,
    'val_split' : 0.15,
    'lstm_do' : 0.1,
    'dense_do' : 0.5,
    'lr' : 0.0005}

possible_answers = np.array(pickle.load(open('answers.p','rb')))

q_types_adv = pickle.load(open('qt_advancedval.p','rb'))
q_ids = pickle.load(open('qidsval.p','rb'))

# Required for hard question type constraints
colour_indexes = pickle.load(open('colour_indexes.p','rb'))
number_indexes = pickle.load(open('number_indexes.p','rb'))
yn_indexes = [0,1]

# Masks for appying hard constraints
yn_mask = np.zeros([1000,])
yn_mask[[0,1]] = 1
n_mask = np.zeros([1000,])
n_mask[number_indexes] = 1
c_mask = np.zeros([1000,])
c_mask[colour_indexes] = 1


loadfile = 'noim_model_save.hdf5'
model = Answerer(params, loadfile)
model.remove_im(params)
model.load_weights(loadfile)
results = []
print('loading')


# Comment out if model needs the inputs
# q_types = pickle.load(open('qtypesval_save.p','rb'))
# qt_advanced = pickle.load(open('qt_advancedval.p','rb'))

# q_types = to_categorical(q_types, 65)
# qt_advanced = to_categorical(qt_advanced, 5)


# feed_in = [pickle.load(open('embsval.p','rb')), pickle.load(open('imsval.p','rb')),q_types, qt_advanced]
# feed_in = [pickle.load(open('embsval.p','rb')), pickle.load(open('imsval.p','rb'))]
feed_in = [pickle.load(open('embsval.p','rb'))]

print('predicting')
  
predictions = model.predict(feed_in)

print('predicted')

# Uncomment if you want to set hard question type constraints
# mask = np.ones_like(predictions)
# mask[np.where(q_types_adv == 0)] = n_mask
# mask[np.where(q_types_adv == 1)] = yn_mask
# mask[np.where(q_types_adv == 3)] = c_mask

# predictions *= mask

predicted_answers = possible_answers[np.argmax(predictions,axis = 1)]
[results.append({'question_id' : q_id, 'answer' : pred}) for q_id, pred in zip(q_ids,predicted_answers)]

# Save the results 
print('dumping')
with open('VQA/Results/v2_OpenEnded_mscoco_val2014_rdiffwidelongdohcni_results.json', 'w') as outfile:
    json.dump(results, outfile)

# model.predict(question_embedding[None,:,:])