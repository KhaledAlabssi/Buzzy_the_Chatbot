import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build(input, output):
    model = Sequential()
    model.add(Dense(128, input_shape=(input,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output, activation='softmax'))
    return model

def train(model, trining_x, trining_y):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(trining_x, trining_y, epochs=200, batch_size=5, verbose=1)
    model.save('chatbot_model.h5')
    return model, history