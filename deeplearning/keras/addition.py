#!/usr/bin/env python3

from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range

DIGITS = 3
LEN_QUESTION = DIGITS * 2 + 1
LEN_ANSWER = DIGITS + 1

def generate_random_number():
    return np.random.randint(0, pow(10, DIGITS))

# def pad_q(s):
#     return pad(s, LEN_QUESTION)
# def pad_a(s):
#     return pad(s, LEN_ANSWER)
def pad(s, l):
    return s + ' ' * (l - len(s))

TRAINING_SIZE = 50000

def generate_data():
    """Generate data
    """
    data = {}
    while len(data) < TRAINING_SIZE:
        a = generate_random_number()
        b = generate_random_number()
        res = a + b
        question = str(a) + '+' + str(b)
        answer = str(res)
        data[question] = answer
    # [(x, ...), (y, ...)]
    x = list(data.keys())
    y = list(data.values())
    return x, y

# put ' ' at the beginning so that encode ' ' as 0,0,0,... TODO This
# need not be the case.
CHARS = ' 1234567890+'
c2i = dict((c, i) for i, c in enumerate(CHARS))
i2c = dict((i, c) for i, c in enumerate(CHARS))

def encode_char(c):
    res = np.zeros(len(CHARS))
    res[c2i[c]] = 1
    return res
def encode(s, l):
    return np.array([encode_char(c) for c in pad(s, l)])
def decode(x, cal_argmax=True):
    if cal_argmax:
        x = x.argmax(axis=-1)
    return ''.join(i2c[xi] for xi in x)

def test():
    encode('12+3', 7)
    decode(encode('12+3', 7))
    return

HIDDEN_SIZE = 128
BATCH_SIZE = 128

def build_model():
    model = Sequential()
    # FIXME how to decide these layers?
    model.add(layers.LSTM(HIDDEN_SIZE, input_shape=(LEN_QUESTION, len(CHARS))))
    model.add(layers.RepeatVector(LEN_ANSWER))
    model.add(layers.LSTM(HIDDEN_SIZE, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(len(CHARS), activation='softmax')))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def train(model, x_train, y_train, x_val, y_val):
    for iteration in range(1, 50):
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1,
                  validation_data=(x_val, y_val))
        for i in range(10):
            index = np.random.randint(0, len(x_val))
            x = x_val[index]
            y = y_val[index]
            # FIXME why must index by an array?
            #
            # FIXME and get the 0 index here
            #
            # These two are equal:
            #
            # > model.predict_classes(vx_val[[0]])
            # > model.predict(vx_val[[0]]).argmax(axis=-1)
            #
            # So, just use predict_classes
            predict = model.predict_classes(x_val[[index]])[0]
            str_x = decode(x)
            str_y = decode(y)
            # must NOT doing the argmax again, because softmax already
            # does that?
            str_predict = decode(predict, cal_argmax=False)
            print('Q:', str_x, 'G:', str_y, 'P:', str_predict,
                  '' if str_y == str_predict else 'X')

def test_model(model, question):
    x = np.array([encode(question, LEN_QUESTION)])
    predict = model.predict_classes(x)[0]
    return decode(predict, cal_argmax=False)

def main():
    x, y = generate_data()
    vx = np.array([encode(xi, LEN_QUESTION) for xi in x])
    vy = np.array([encode(yi, LEN_ANSWER) for yi in y])
    model = build_model()
    model.summary()
    split_at = len(vx) // 10
    vx_train = vx[split_at:]
    vx_val = vx[:split_at]
    vy_train = vy[split_at:]
    vy_val = vy[:split_at]
    vx_train.shape
    vy_train.shape
    vx_val.shape
    vy_val.shape
    train(model, vx_train, vy_train, vx_val, vy_val)
    
    # Q: 320+811 G: 1131 P: 1141
    # Q: 90+185  G: 275  P: 266 
    # Q: 948+862 G: 1810 P: 1810
    # Q: 700+548 G: 1248 P: 1248
    # Q: 895+373 G: 1268 P: 1268
    # Q: 698+550 G: 1248 P: 1248
    # Q: 993+809 G: 1802 P: 1802
    test_model(model, '100+28')
    test_model(model, '320+811')
    test_model(model, '90+185')
    test_model(model, '873+1')
    test_model(model, '666+3')
    test_model(model, '70+185')

