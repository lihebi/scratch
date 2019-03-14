import logging
import os
import numpy as np
import tensorflow as tf

from cleverhans.attacks import CarliniWagnerL2
from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.utils import grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_eval, tf_model_load
from cleverhans.train import train
from cleverhans.model_zoo.basic_cnn import ModelBasicCNN

VIZ_ENABLED = True
BATCH_SIZE = 128
NB_EPOCHS = 6
SOURCE_SAMPLES = 10
LEARNING_RATE = .001
CW_LEARNING_RATE = .2
ATTACK_ITERATIONS = 100
MODEL_PATH = os.path.join('models', 'mnist')
TARGETED = True

def __test():
    # report = AccuracyReport()
    tf.set_random_seed(1234)
    sess = tf.Session()
    set_log_level(logging.DEBUG)

    # Get MNIST test data
    mnist = MNIST(train_start=0, train_end=60000,
                  test_start=0, test_end=10000)
    x_train, y_train = mnist.get_set('train')
    x_test, y_test = mnist.get_set('test')
    # Obtain Image Parameters
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]
    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                          nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    nb_filters = 64
    # Define TF model graph
    model = ModelBasicCNN('model1', nb_classes, nb_filters)
    preds = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=0.1)
    print("Defined TensorFlow model graph.")
    # Train an MNIST model
    train_params = {
        'nb_epochs': NB_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'filename': os.path.split(MODEL_PATH)[-1]
    }

    rng = np.random.RandomState([2017, 8, 30])
    # check if we've trained before, and if we have, use that pre-trained model
    if os.path.exists(model_path + ".meta"):
        tf_model_load(sess, model_path)
    else:
        train(sess, loss, x_train, y_train, args=train_params, rng=rng)
        saver = tf.train.Saver()
        saver.save(sess, model_path)
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': BATCH_SIZE}
    accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
    assert x_test.shape[0] == test_end - test_start, x_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    # report.clean_train_clean_eval = accuracy
