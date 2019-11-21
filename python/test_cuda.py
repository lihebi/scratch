import keras
import tensorflow as tf
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time

from utils import *
from tf_utils import *

def test():

    sess = create_tf_session()
    
    inputs = keras.layers.Input(shape=(28,28,1), dtype='float32')
    x = inputs
    # x = keras.layers.Reshape(self.xshape())(inputs)
    x = keras.layers.Conv2D(32, 5)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPool2D((2,2))(x)
    x = keras.layers.Conv2D(64, 5)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPool2D((2,2))(x)

    # inputs = keras.layers.Input(batch_shape=shape, dtype='float32')
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1024)(x)
    x = keras.layers.Activation('relu')(x)
    logits = keras.layers.Dense(10)(x)
    model = keras.models.Model(inputs, logits)
    y = keras.layers.Input(shape=(10,), dtype='float32')
    
    loss = my_softmax_xent(logits, y)
    accuracy = my_accuracy_wrapper(logits, y)

    tf_init_uninitialized(sess)
    
    def myloss(ytrue, ypred):
        return loss
    def acc(ytrue, ypred): return accuracy

    (train_x, train_y), (test_x, test_y) = load_mnist_data()

    with sess.as_default():
        model.compile(loss=myloss,
                      metrics=[acc],
                      optimizer=keras.optimizers.Adam(lr=1e-3),
                      target_tensors=y)
        model.fit(train_x, train_y,
                  batch_size=32,
                  shuffle=True,
                  validation_split=0.1,
                  epochs=10)

if __name__ == '__main__':
    test()
