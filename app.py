from preparation import get_data, preprocessing, finalize_data
from training import build, train
from util import predict_class, get_response
import pickle
import json


if __name__ == "__main__":
    data = get_data('intents.json')
    trining_x, trining_y, words, classes = preprocessing(data)
    finalize_data(words, classes)

    model = build(len(trining_x[0]), len(trining_y[0]))
    model, history = train(model, trining_x, trining_y)

    find_class = predict_class("hello", model, words, classes)
    answer = get_response(find_class, data)
    print("Buzzy bot's Anser: ", answer)


