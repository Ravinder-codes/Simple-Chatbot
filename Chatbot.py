import json
import pickle
from tensorflow import keras
from nltk import word_tokenize as token
from nltk.stem import WordNetLemmatizer
import numpy as np
import random

# Loading data from files
intents = json.loads(open('intents.json').read())
words_set = pickle.load(open('words.pkl','rb'))
tags_list = pickle.load(open('tags.pkl','rb'))
model = keras.models.load_model('ChatbotModel.h5')

lemma = WordNetLemmatizer()

# Function to clean up user sentence
def CleanSentence(query):
    query_words = token(query)
    query_words = [lemma.lemmatize(word.lower()) for word in query_words]
    return query_words

# Creating a dataset for the given sentence
def ConvertQueryToArray(query):
    query_words = CleanSentence(query)
    bag_ofwords = [0]*len(words_set)

    for i,word in enumerate(words_set):
        if(word in query_words):bag_ofwords[i] = 1
    return np.array(bag_ofwords)


# Get the most accurate tag
def Predict_Reply(query):
    bag = ConvertQueryToArray(query)
    prediction = list(model.predict(np.array([bag]))[0])
    
    reply_tag  = tags_list[prediction.index(max(prediction))]
    return reply_tag


# Bot replies to the asked questions based on the predictions
sentence = ""
print("------------------ANY QUESTIONS REGARDING UNIVERITY? ASK CHATBOT ------------------")

while(sentence!="quit"):
    sentence = input(">>>")
    tag = Predict_Reply(sentence)

    for intent in intents["intents"]:
        if(intent['tag']==tag):
            print("BOT :",random.choice(intent['response']))
            break