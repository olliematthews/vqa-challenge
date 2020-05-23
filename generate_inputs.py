# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:43:29 2020

File to run through the training or validation data and generate
what is needed to train the models used. Note that it should be 
run for train first to generate the question types

Everything is saved as a pickle file for loading

We output:
    the embeddings (from spacy)
    the image features
    the answer indexes
    the question type
    the more advanced question type (e.g. closed etc.):
        0 -> number question
        1 -> yes/no question
        2 -> or question
        3 -> colour question
        4 -> other
        
@author: Ollie
"""

import numpy as np
import json
import pickle
from pathlib import Path
import spacy

def load_json(file_name):
    # Takes the file name as an input and loads it
    path = Path('../QAs/' + file_name + '.json')
    with open(path) as file:
        data = json.load(file)
    if 'questions' in data.keys():
        return data['questions']
    else:
        return data['annotations']

dataset = 'train'
nlp = spacy.load("en_core_web_sm")
quests = load_json('Q_' + dataset)
anns = load_json('A_' + dataset)
features = pickle.load(open('../features/' + dataset + '.pickle','rb'))
possible_answers = pickle.load(open('../answers.p','rb'))


embedding_vector_space = np.zeros([24,96])
question_embeddings = []
embedding_vectors = []
image_features = []
q_types = []
q_advanceds = []

closed_words = ['is','are','has', 'could', 'do']
open_words = ['what', 'which', 'why', 'where', 'how', 'who']


if dataset == 'train':
    answers = []
    q_type_array = []

else:
    qids = []   
    q_type_array = pickle.load('possible_q_types.p','rb')

i = 0
for question, annotation in zip(quests,anns):
    i += 1
    assert question['question_id'] == annotation['question_id']
    
    # If we are in the training set, we ignore questions with unrecog-
    # nised answers
    if dataset == 'train':
        if not annotation['multiple_choice_answer'] in possible_answers:
            continue
        
        
    image_features.append(features[annotation['image_id']])

    question_embedding = nlp(question['question'])
    len_emb = len(question_embedding)
    
    embedding_vector = embedding_vector_space.copy()
    
    # If embedding is too long, truncate. If it is too short, pad.
    if len_emb > 24:
        embedding_vector[:, :] = np.array([token.vector for token in question_embedding])[:24]
    else:
        embedding_vector[:len_emb, :] = np.array([token.vector for token in question_embedding])
        
    embedding_vectors.append(embedding_vector)
    
    if dataset == 'val':
        qids.append(question['question_id'])
    else:
        answers.append(possible_answers.index(annotation['multiple_choice_answer']))

   
    # Create question type array. Can comment out if not needed.
    if not annotation['question_type'] in q_type_array:
        assert dataset == 'train'
        q_type_array.append(annotation['question_type'])
        
    if 'many' in annotation['question_type'] or 'number' in annotation['question_type']:
        q_advanceds.append(0)
    elif np.any([word in annotation['question_type'] for word in closed_words]):
        if np.all([not word in annotation['question_type'] for word in open_words]):
            if not 'or' in question['question']:
                q_advanceds.append(1)
            else:
                q_advanceds.append(2)
        elif 'color' in annotation['question_type']:
            q_advanceds.append(3)
        else:
            q_advanceds.append(4)
    elif 'color' in annotation['question_type']:
        q_advanceds.append(3)
    else:
        q_advanceds.append(4)
        
    q_types.append(q_type_array.index(annotation['question_type']))

    # Print progress from time to time
    if i % 2000 == 0:
        print(str(np.round(100 * i / len(anns), )) + '% done')
    
embedding_vectors = np.array(embedding_vectors)
image_features = np.array(image_features)
answers = np.array(answers)


# Save everything
pickle.dump(embedding_vectors, open('embs' + dataset  '.p','wb'))
pickle.dump(image_features, open('ims' + dataset  '.p','wb'))
pickle.dump(answers, open('ans' + dataset  '.p','wb'))

pickle.dump(qids, open('qids' + dataset '.p','wb'))
pickle.dump(q_types,open('qtypes' + dataset + '.p','wb'))
pickle.dump(q_advanceds, open('qt_advanced' + dataset +  '.p','wb'))

if dataset == 'train':
    pickle.dump(q_type_array, open('possible_q_types.p','wb'))

