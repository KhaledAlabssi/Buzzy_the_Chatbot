import json
import nltk
from nltk.stem import WordNetLemmatizer

# load data
with open('intents.json') as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()

# initialize lists
words = []
classes = []
documents = []


# lemmatize & tokenize

for intent in data['intents']:

    for pattern in intent['patterns']:

        wordList = nltk.word.tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

