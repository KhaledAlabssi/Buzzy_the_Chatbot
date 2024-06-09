import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
 
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
