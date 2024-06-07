import json
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle


# load data
with open('intents.json') as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()

# initialize lists
words = []
classes = []
documents = []


# tokenize

for intent in data['intents']:

    for pattern in intent['patterns']:

        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize

ignore_words = ["?", "!", ".", ", "]
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

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

# model: 
model = Sequential()
model.add(Dense(128, input_shape=(len(trining_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(trining_y[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(trining_x, trining_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
