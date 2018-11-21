#!/usr/bin/env python3

import numpy as np
import os
import tempfile

import keras
from keras import backend as K
from keras import layers
from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import tensorflow as tf
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
# epochs = 12
epochs = 5

# input image dimensions
nrows, ncols = 28, 28
input_shape = (nrows, ncols, 1)

# (insert-image (create-image "/etc/alternatives/emacs-128x128.png"))

def visualize(data):
    plt.imshow(data, cmap='Greys')
    # plt.show()

    # Save to a file:
    # pylab.ioff()
    # plot([1, 2, 3])
    # savefig("/tmp/test.png")
    return

def test_visualize():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    image_index = 7777 # You may select anything up to 60,000
    print(y_train[image_index]) # The label is 8
    image_data = x_train[image_index]
    visualize(image_data)
    return
    

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

def prepare_data_transfer():
    # (HEBI: num_classes now is 5)
    num_classes = 5
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(x_train.shape[0], nrows, ncols, 1)
    x_test = x_test.reshape(x_test.shape[0], nrows, ncols, 1)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # (HEBI: split the data)
    x_train_lt5 = x_train[y_train<5]
    x_test_lt5 = x_test[y_test<5]
    y_train_lt5 = y_train[y_train<5]
    y_test_lt5 = y_test[y_test<5]
    
    x_train_gte5 = x_train[y_train>=5]
    x_test_gte5 = x_test[y_test>=5]
    # (HEBI: -5), otherwise the to_categorical will throw error.
    y_train_gte5 = y_train[y_train>=5] - 5
    y_test_gte5 = y_test[y_test>=5] - 5
    
    # process labels
    y_train_lt5 = keras.utils.to_categorical(y_train_lt5, num_classes)
    y_train_gte5 = keras.utils.to_categorical(y_train_gte5, num_classes)
    y_test_lt5 = keras.utils.to_categorical(y_test_lt5, num_classes)
    y_test_gte5 = keras.utils.to_categorical(y_test_gte5, num_classes)
    lt5 = (x_train_lt5, y_train_lt5), (x_test_lt5, y_test_lt5)
    gte5 = (x_train_gte5, y_train_gte5), (x_test_gte5, y_test_gte5)
    return lt5, gte5

def train(model, x_train, y_train, x_test, y_test):
    # fit
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return

def train_1(model, data):
    (x_train, y_train), (x_test, y_test) = data
    train_4(model, x_train, y_train, x_test, y_test)
    return

def train_2(model, train_data, test_data):
    x_train, y_train = train_data
    x_test, y_test = test_data
    train(model, x_train, y_train, x_test, y_test)
    return
    

def transfer_model():
    # TODO should not define here
    num_classes = 5
    # define two groups of layers: feature (convolutions) and classification (dense)
    feature_layers = [
        # FIXME kernel size 3 or (3,3)?
        Conv2D(32, kernel_size=(3,3),
               padding='valid',
               input_shape=input_shape),
        Activation('relu'),
        Conv2D(32, kernel_size=(3,3)),
        Activation('relu'),
        # FIXME 2 or (2,2)
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        Flatten()]

    classification_layers = [
        Dense(128),
        Activation('relu'),
        Dropout(0.5),
        Dense(num_classes),
        Activation('softmax')]
    # create complete model
    model = Sequential(feature_layers + classification_layers)
    def freeze_feature():
        # freeze feature layers and rebuild model
        for l in feature_layers:
            l.trainable = False
    model.freeze_feature = freeze_feature
    return model
    
def transfer():
    lt5, gte5 = prepare_data_transfer()
    model = transfer_model()
    # train model for 5-digit classification [0..4]
    train_1(model, lt5)
    # transfer: train dense layers for new classification task [5..9]
    model.freeze_feature()
    train_1(model, gte5)
    return

def prepare_data_flat():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # using 784 instead of 28,28,1
    x_train = x_train.reshape(x_train.shape[0], nrows * ncols)
    x_test = x_test.reshape(x_test.shape[0], nrows * ncols)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # process labels
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
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

def mlp_model():
    model = Sequential()
    # this require the input shape to be 784
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model


def cnn_model():
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

def hrnn_model():
    """Hierarchical recurrent neural network.

    This seems to be very slow. TODO why? RNN? Also, why it is
    hierarchy?

    """
    row_hidden = 128
    col_hidden = 128
    x = Input(shape=(nrows, ncols, 1))
    # Encodes a row of pixels using TimeDistributed Wrapper.
    encoded_rows = TimeDistributed(LSTM(row_hidden))(x)
    # Encodes columns of encoded rows.
    encoded_columns = LSTM(col_hidden)(encoded_rows)
    # Final predictions and model.
    prediction = Dense(num_classes, activation='softmax')(encoded_columns)
    model = Model(x, prediction)
    return model

def irnn_model():
    """Seems to require 900 epochs to reach 0.93 accuracy.
    
    http://arxiv.org/pdf/1504.00941v2.pdf
    """
    model = Sequential()
    model.add(SimpleRNN(hidden_units,
                        kernel_initializer=initializers.RandomNormal(stddev=0.001),
                        recurrent_initializer=initializers.Identity(gain=1.0),
                        activation='relu',
                        input_shape=x_train.shape[1:]))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
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
    # either one
    (x_train, y_train), (x_test, y_test) = prepare_data()
    (x_train, y_train), (x_test, y_test) = prepare_data_flat()

    # models
    model = cnn_model()
    model = irnn_model()
    model = hrnn_model()
    model = mlp_model()

    # fit
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])

    # We save the weights to a file, and load it back.
    weight_path = os.path.join(tempfile.gettempdir(), 'saved_wt.h5')
    weight_path
    model.save_weights(weight_path)
    # Clean up the TF session.
    K.clear_session()
    test_model = cnn_model()
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
    
