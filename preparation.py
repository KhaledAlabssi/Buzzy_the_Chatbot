import json
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle

def get_data(data):
    with open(data) as file:
        return json.load(file)

def preprocessing(data):
    lemmatizer = WordNetLemmatizer()
    words = []
    classes = []
    documents = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wordList = nltk.word_tokenize(pattern)
            words.extend(wordList)
            documents.append((wordList, intent['tag']))

            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    
    ignore = ["?", "!", ".", ", "]
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore]
    
    words = sorted(set(words))
    classes = sorted(set(classes))
    # print(f"words: {words}")
    # print(f"classes: {classes}")

    training = []
    outputEmpty = [0] * len(classes)

    for doc in documents:
        contain = []
        patternWords = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
        for word in words:
            contain.append(1) if word in patternWords else contain.append(0)
        
        output = list (outputEmpty)
        output[classes.index(doc[1])] = 1
        training.append([contain, output])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    trining_x = np.array(list(training[:, 0]))
    trining_y = np.array(list(training[:, 1]))

    return trining_x, trining_y, words, classes

def finalize_data(words, classes):
    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))