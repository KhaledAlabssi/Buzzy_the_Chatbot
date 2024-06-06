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


