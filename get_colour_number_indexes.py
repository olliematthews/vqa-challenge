# -*- coding: utf-8 -*-
'''
Script to find the colours and numbers out of the possible answers
and save their indexes
'''

import pickle


possible_answers = pickle.load(open('answers.p','rb'))
colour_indexes = []
with open('colors.txt', 'r') as file:
    for line in file:
        colour = line.split('\n')[0].lower()
        if colour in possible_answers:
            index = possible_answers.index(colour)
            colour_indexes.append(index)
            

number_indexes = []
for i, ans in enumerate(possible_answers):
    try:
        int(ans)
        number_indexes.append(i)
    except:
        pass
    
    
pickle.dump(colour_indexes, open('colour_indexes.p','wb'))
pickle.dump(number_indexes, open('number_indexes.p','wb'))
           
            