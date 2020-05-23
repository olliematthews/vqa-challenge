# -*- coding: utf-8 -*-
"""
The model used for classification.
"""

import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Flatten, Multiply, Input, Dense, Concatenate, Dropout
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.callbacks.callbacks import ModelCheckpoint

class Answerer:
    def __init__(self, params, savefile):
        self.batch_size = params['batch_size']
        embeddings = Input(shape = (24,96))
        image_features = Input(shape = (2048,))
        
        
        lstm_layers = []
        lstm_dropout = []
        layers = [embeddings]
        for lstm_layer_width in params['lstm_layers']:
            lstm_layers.append(LSTM(lstm_layer_width, return_sequences=True, dropout = params['lstm_do'])(layers[-1]))
            layers.append(lstm_layers[-1])

        lstm_out = Flatten()(layers[-1])
        
        image_out = Dense(params['lstm_layers'][-1] * 24, activation = 'relu')(image_features)

        image_out_dropout = Dropout(params['dense_do'])(image_out)
        multiply = Multiply()([lstm_out, image_out_dropout])

        dense_layers = []
        dense_dropout = []
        layers = [multiply]
        for dense_layer_width in params['dense_layers']:
            dense_layers.append(Dense(dense_layer_width, activation = 'relu')(layers[-1]))
            dense_dropout.append(Dropout(params['dense_do'])(dense_layers[-1]))
            layers.append(dense_dropout[-1])
        output = Dense(params['out_dimensionality'], activation = 'softmax')(layers[-1])
        
        self.model = Model(inputs = [embeddings, image_features], outputs = output)
        self.model.compile(Adam(learning_rate = params['lr']), loss = sparse_categorical_crossentropy)
        self.model.summary()
        self.callback = ModelCheckpoint(savefile, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)
        self.val_split = params['val_split']

    def load_weights(self, savefile):
        self.model.load_weights(savefile)
        
    def learn(self, qas, epochs):
        self.model.fit(qas[0], qas[1], epochs = epochs, batch_size = self.batch_size, callbacks = [self.callback], validation_split = self.val_split, shuffle = True)

    def predict(self, inputs):
        return self.model.predict(inputs)
    
    
    
    '''
    The following functions are used to take a previously defined
    model and alter it while saving the weights:
        The first adds a concatenation layer to input the question type 
        The second adds FCC layers on the end
        The third removes the image layer to test the LSTM arm alone  
    '''
    def add_qtype(self, params):
        # Add question type as an input.
        
        # Save current weights
        weights = []
        for layer in self.model.layers[:-1]:
            weights.append(layer.get_weights())
        
        # Rebuild network with concatenation layer
        embeddings = Input(shape = (24,96))
        image_features = Input(shape = (2048,))
        q_typein = Input(shape = (65,))
        qt_advancedin = Input(shape = (5,))

        lstm_layers = []
        lstm_dropout = []
        layers = [embeddings]
        for lstm_layer_width in params['lstm_layers']:
            lstm_layers.append(LSTM(lstm_layer_width, return_sequences=True, dropout = params['lstm_do'])(layers[-1]))
            layers.append(lstm_layers[-1])

        lstm_out = Flatten()(layers[-1])
        image_out = Dense(params['lstm_layers'][-1] * 24, activation = 'relu')(image_features)

        image_out_dropout = Dropout(params['dense_do'])(image_out)
        multiply = Multiply()([lstm_out, image_out_dropout])
        conc = Concatenate()([multiply, q_typein, qt_advancedin])
        dense_layers = []
        dense_dropout = []
        layers = [conc]
        for dense_layer_width in params['dense_layers']:
            dense_layers.append(Dense(dense_layer_width, activation = 'relu')(layers[-1]))
            dense_dropout.append(Dropout(params['dense_do'])(dense_layers[-1]))
            layers.append(dense_dropout[-1])
        
        output = Dense(params['out_dimensionality'], activation = 'softmax')(layers[-1])
                
        # Compile
        self.model = Model(inputs = [embeddings, image_features, q_typein, qt_advancedin], outputs = output)
        self.model.compile(Adam(learning_rate = params['lr']), loss = sparse_categorical_crossentropy)
        self.model.summary()
        
        # Reload weights
        for layer, weight in zip(self.model.layers, weights):
            layer.set_weights(weight)
            
    def add_layer(self, params):
        # Adds FCC layer on the end
        
        # Save weights
        weights = []
        for layer in self.model.layers[:-1]:
            weights.append(layer.get_weights())

        # Rebuild model
        embeddings = Input(shape = (24,96))
        image_features = Input(shape = (2048,))
        
        
        lstm_layers = []
        lstm_dropout = []
        layers = [embeddings]
        for lstm_layer_width in params['lstm_layers']:
            lstm_layers.append(LSTM(lstm_layer_width, return_sequences=True, dropout = params['lstm_do'])(layers[-1]))
            layers.append(lstm_layers[-1])

        lstm_out = Flatten()(layers[-1])
        
        image_out = Dense(params['lstm_layers'][-1] * 24, activation = 'relu')(image_features)

        image_out_dropout = Dropout(params['dense_do'])(image_out)
        multiply = Multiply()([lstm_out, image_out_dropout])

        dense_layers = []
        dense_dropout = []
        layers = [multiply]
        for dense_layer_width in params['dense_layers']:
            dense_layers.append(Dense(dense_layer_width, activation = 'relu')(layers[-1]))
            dense_dropout.append(Dropout(params['dense_do'])(dense_layers[-1]))
            layers.append(dense_dropout[-1])
        output = Dense(params['out_dimensionality'], activation = 'softmax')(layers[-1])
        
        self.model = Model(inputs = [embeddings, image_features], outputs = output)
        self.model.compile(Adam(learning_rate = params['lr']), loss = sparse_categorical_crossentropy)
        self.model.summary()
        
        # Reload weights
        for layer, weight in zip(self.model.layers, weights):
            layer.set_weights(weight)

    def remove_im(self, params):
        # Removes the image branch. Be careful, it will not work for any model
        # architechture. Assumes two lstm cells and one dense layer w/ dropout
        # on the image features.
        
        # Save weights
        weights = []
        for i, layer in enumerate(self.model.layers):
            if i == 2 or i == 4 or i == 6 or i == 7:
                continue
            else:
                weights.append(layer.get_weights())
        
        # Rebuild model
        embeddings = Input(shape = (24,96))

        lstm_layers = []
        lstm_dropout = []
        layers = [embeddings]
        for lstm_layer_width in params['lstm_layers']:
            lstm_layers.append(LSTM(lstm_layer_width, return_sequences=True, dropout = params['lstm_do'])(layers[-1]))
            layers.append(lstm_layers[-1])

        lstm_out = Flatten()(layers[-1])
    

        dense_layers = []
        dense_dropout = []
        layers = [lstm_out]
        for dense_layer_width in params['dense_layers']:
            dense_layers.append(Dense(dense_layer_width, activation = 'relu')(layers[-1]))
            dense_dropout.append(Dropout(params['dense_do'])(dense_layers[-1]))
            layers.append(dense_dropout[-1])
        
        output = Dense(params['out_dimensionality'], activation = 'softmax')(layers[-1])
                
        self.model = Model(inputs = [embeddings], outputs = output)
        self.model.compile(Adam(learning_rate = params['lr']), loss = sparse_categorical_crossentropy)
        self.model.summary()
        
        # Reload weights
        for layer, weight in zip(self.model.layers, weights):
            layer.set_weights(weight)
            


