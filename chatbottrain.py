# PROJECT BY RAVINDER SINGH
import numpy as np
from nltk import word_tokenize as token
import json
import pickle
from nltk.stem import WordNetLemmatizer
import random
from tensorflow import keras

# Loading intents
data_intents = json.loads(open('intents.json').read())
words_set=set()
training_list=[]
tags_list=[]
ignore = ['!','.','/','?',';',',',':',':']

lemma = WordNetLemmatizer()

# Fetching data and updating lists accordingly
for intent in data_intents['intents']:
    tags_list.append(intent['tag'])
    for pattern in intent['patterns']:
        # Tokenizing and lemmatizing the words
        pattern_words = token(pattern)
        pattern_words= [lemma.lemmatize(word.lower()) for word in pattern_words if word not in ignore]
        
        words_set.update(pattern_words)
        training_list.append([pattern_words,intent['tag']])
    
words_set = list(words_set)

# Saving the cleaned data in file
pickle.dump(words_set,open('words.pkl','wb'))
pickle.dump(tags_list,open('tags.pkl','wb'))


# Creating a training set
wordslen = len(words_set)
tagslen = len(tags_list)
for i in range(len(training_list)):
    train_data = [0]*wordslen
    output = [0]*tagslen

    # Setting value 1 where the word belongs to the pattern
    for j in range(wordslen):
        if(words_set[j] in training_list[i][0]):train_data[j]=1
    
    # Stores to which tag the pattern belongs
    output[tags_list.index(training_list[i][1])]= 1
    training_list[i]= train_data+output

random.shuffle(training_list)
training_list = np.array(training_list)
trainX = training_list[:,:len(words_set)]
trainY = training_list[:,len(words_set):]

# Creating a neural model
model = keras.Sequential()
model.add(keras.layers.Dense(128,input_shape=(len(trainX[0]),),activation = 'relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dense(32,activation='relu'))
model.add(keras.layers.Dense(len(trainY[0]),activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
record = model.fit(trainX,trainY,epochs = 100,verbose=1)
model.save('ChatbotModel.h5',record)