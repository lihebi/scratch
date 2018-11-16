#!/usr/bin/env python3
import tensorflow as tf

def test_tf():
    import tensorflow as tf
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
      raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))
    
    hello = tf.constant('hello')
    sess = tf.Session()
    print(sess.run(hello))


def test_keras():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)
    return

def tf_cpu_gpu_comparison():
    import timeit

    # See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.device('/cpu:0'):
      random_image_cpu = tf.random_normal((100, 100, 100, 3))
      net_cpu = tf.layers.conv2d(random_image_cpu, 32, 7)
      net_cpu = tf.reduce_sum(net_cpu)

    with tf.device('/gpu:0'):
      random_image_gpu = tf.random_normal((100, 100, 100, 3))
      net_gpu = tf.layers.conv2d(random_image_gpu, 32, 7)
      net_gpu = tf.reduce_sum(net_gpu)

    sess = tf.Session(config=config)

    # Test execution once to detect errors early.
    try:
      sess.run(tf.global_variables_initializer())
    except tf.errors.InvalidArgumentError:
      print(
          '\n\nThis error most likely means that this notebook is not '
          'configured to use a GPU.  Change this in Notebook Settings via the '
          'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
      raise

    def cpu():
      sess.run(net_cpu)

    def gpu():
      sess.run(net_gpu)

    # Runs the op several times.
    print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
          '(batch x height x width x channel). Sum of ten runs.')
    print('CPU (s):')
    cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
    print(cpu_time)
    print('GPU (s):')
    gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
    print(gpu_time)
    print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))

    sess.close()

def main():
    pass
