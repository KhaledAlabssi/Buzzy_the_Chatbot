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


def clean_sentence(sentence):
    s_words = nltk.word_tokenize(sentence)
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]
    return s_words

def bow(sentence, words):
    s_words = clean_sentence(sentence)
    container = [0] * len(words)
    for s in s_words:
        for i, w in enumerate(words):
            if w == s:
                container[i] = 1
    return np.array(container)

def predict_class (sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    err_threshold = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > err_threshold]
    result.sort(key = lambda x: x[1], reverse=True)
    res_list = []
    for r in result:
        res_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return res_list


def get_response(intent_list, intents):
    label = intent_list[0]['intent']
    intents_collection = intents['intents']
    for i in intents_collection:
        if i['tag'] == label:
            result = random.choice(i['responses'])
            break
    return result


find_class = predict_class("hello", model)
answer = get_response(find_class, data)
print("Answer: ", answer)
