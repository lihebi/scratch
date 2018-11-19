#!/usr/bin/env python3

import numpy as np
import os
import tempfile

import keras
from keras import backend as K
from keras import layers
from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import tensorflow as tf

batch_size = 128
num_classes = 10
# epochs = 12
epochs = 5

# input image dimensions
nrows, ncols = 28, 28
input_shape = (nrows, ncols, 1)

def prepare_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # A little analysis of mnist data
    x_train.shape               # (60000, 28, 28)
    y_train.shape               # (60000,)
    x_test.shape                # (10000, 28, 28)
    y_test.shape                # (10000,)
    # FIXME K.image_data_format()
    # channels_last
    # FIXME WHY
    x_train = x_train.reshape(x_train.shape[0], nrows, ncols, 1)
    x_train.shape               # (60000, 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], nrows, ncols, 1)
    # the input are 0-255. change to float
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # process labels
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    # previously, the label data is 0-9. Now, it is a one-hot
    # vector. The num_classes can be omitted, because we have 10
    # values here. We can supply a number >10, but not below (error
    # otherwise).
    y_train.shape               # (60000, 10)
    y_test.shape
    return (x_train, y_train), (x_test, y_test)

def prepare_data_2():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train.shape               # (60000, 28, 28)
    y_train.shape               # (60000,)
    x_test.shape
    y_test.shape
    x_train = x_train.astype(np.float32) / 255
    # this will achive the same effect as reshape
    x_train = np.expand_dims(x_train, -1)
    x_train.shape               # (60000, 28, 28, 1)
    y_train = tf.one_hot(y_train, num_classes)
    # TensorShape([Dimension(60000), Dimension(10)]), should be same
    # as (60000, 10)
    y_train.shape
    # FIXME not dividing 255?
    x_test = x_test.astype(np.float32)
    x_test = np.expand_dims(x_test, -1)
    x_test.shape                # (10000, 28, 28, 1)
    # If we omit the convertion for y_test, during evaluating, we need
    # to use:
    #
    # >>> model.evaluate(x_test, y_test, num_classes)
    y_test = y_test.astype(np.float32)
    y_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def test_database_api():
    """I don't like this method.
    """
    def cnn_layers(inputs):
        x = layers.Conv2D(32, (3, 3),
                          activation='relu', padding='valid')(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        predictions = layers.Dense(num_classes,
                                   activation='softmax',
                                   name='x_train_out')(x)
        return predictions

    batch_size = 128
    buffer_size = 10000
    steps_per_epoch = int(np.ceil(60000 / float(batch_size)))  # = 469
    epochs = 5
    num_classes = 10
    def train_using_dataset(x_train, y_train):
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        inputs, targets = iterator.get_next()
        # input is created with the data, instead of random placeholders
        model_input = layers.Input(tensor=inputs)
        model_output = cnn_layers(model_input)
        # This model is created by feeding input and output
        model = keras.models.Model(inputs=model_input, outputs=model_output)
        model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-3, decay=1e-5),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'],
                      target_tensors=[targets])
        model.summary()
        model.fit(epochs=epochs,
                  steps_per_epoch=steps_per_epoch)
        return model

def main():
    (x_train, y_train), (x_test, y_test) = prepare_data()
    model = build_model()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # We save the weights to a file, and load it back.
    weight_path = os.path.join(tempfile.gettempdir(), 'saved_wt.h5')
    weight_path
    model.save_weights(weight_path)
    # Clean up the TF session.
    K.clear_session()
    test_model = build_model()
    test_model.load_weights(weight_path, reshape=True)
    # To use the test_model with loaded weights, not only should the
    # model architecture match, but also the optimizer and loss when
    # COMPILE the model
    #
    # test_model.compile(optimizer='rmsprop',
    #                    loss='sparse_categorical_crossentropy',
    #                    metrics=['accuracy'])
    test_model.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer=keras.optimizers.Adadelta(),
                       metrics=['accuracy'])
    test_model.summary()
    # FIMXE error, the output dim not matching
    loss, acc = test_model.evaluate(x_test, y_test)
    print('\nTest accuracy: {0}'.format(acc))
    
