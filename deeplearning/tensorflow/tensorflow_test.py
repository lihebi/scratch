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

    from tensorflow.python.client import device_lib
    print (device_lib.list_local_devices())
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    1+1
    # (python-shell-send-string "1+1")


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
    # model = tf.keras.utils.multi_gpu_model(model, gpus=2)
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
    return

def test_tensor():
    """To figure out what is tensor

    Available tensor types:
    - tf.Variable
    - tf.constant
    - tf.placeholder
    """
    # Build a dataflow graph.
    c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    e = tf.matmul(c, d)
    c
    x = tf.reduce_sum(e)
    y = tf.negative(x)
    c.shape
    e.shape
    x.shape
    yabs = tf.abs(x)

    # Construct a `Session` to execute the graph.
    sess = tf.Session()

    # Execute the graph and store the value that `e` represents in `result`.
    result = sess.run(e)
    sess.run(x)
    sess.run(e)
    sess.run(y)
    sess.run(yabs)
    # after registering default session, you can use e.eval() to
    # compute the value
    with tf.Session():
        print (e.eval())
    with sess:
        print(e.eval())

    # this produce events.out.tfevents.{timestamp}.{hostname}
    writer = tf.summary.FileWriter('.')
    writer.add_graph(tf.get_default_graph())
    # now, run tensorboard --logdit .

    # of course we can feed in placeholders, instead of constants
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = x + y
    sess.run(z, feed_dict={x: 3, y: 4.5})
    # since feed_dict is the 2nd parameters by position, it can be
    # omitted like this:
    sess.run(z, {x: 3, y: 4.5})
    sess.run(z, feed_dict={x: [1, 3], y: [2, 4]})

    # for more complicated data, we are not using placeholders, but
    # tf.data API
    my_data = [[0, 1,], [2, 3,], [4, 5,], [6, 7,],]
    slices = tf.data.Dataset.from_tensor_slices(my_data)
    next_item = slices.make_one_shot_iterator().get_next()
    # the first four times, we get the data
    sess.run(next_item)
    sess.run(next_item)
    sess.run(next_item)
    sess.run(next_item)
    # this time, throw tf.errors.OutOfRangeError
    sess.run(next_item)
    # or we can see this in the loop
    while True:
      try:
        print(sess.run(next_item))
      except tf.errors.OutOfRangeError:
        break
    # random sample
    r = tf.random_normal([10,3])
    # as expected, each run will produce a different data
    sess.run(r)
    sess.run(r)    
    return

def test_tensor_layer():
    sess = tf.Session()
    # place holder input
    x = tf.placeholder(tf.float32, shape=[None, 3])
    # a linear layer
    linear_model = tf.layers.Dense(units=1)
    y = linear_model(x)
    # initializers
    init = tf.global_variables_initializer()
    # Seems this will initialize all parameters to random values.
    # Each run of initialization will produce different parameters,
    # thus will produce different output for y.
    #
    # If not initializing, the y will be same.
    #
    # If not running initizliation at all, it will through this error:
    #
    # FailedPreconditionError: Attempting to use uninitialized value
    sess.run(init)
    sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]})
    #
    #
    # Now let's train a simple model
    x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
    y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
    # lineear model
    linear_model = tf.layers.Dense(units=1)
    y_pred = linear_model(x)
    # now, we define our loss
    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
    # the optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    # this train is an OPERATOR, not a tensor. It will build all
    # computation graph components necessary for the
    # optimization. When it runs, it will not output its value (it
    # does not have a value). Instead, it will update the variables
    # (weights?) in the graph.
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    # this is a random prediction
    print(sess.run(y_pred))
    # now we run it 100 times
    for i in range(1000):
        # since teh train is an op, not a tensor, it does not have a
        # value. Thus the loss here is for visualizing the progress
        # only.
        _, loss_value = sess.run((train, loss))
        if i % 100 == 0:
            print(loss_value)
    # now, since the graph variables (weights?) are updated, we can
    # get a better prediction
    print(sess.run(y_pred))


def tensor_eager():
    # now is false
    tf.executing_eagerly()
    # this must be called before any tensorflow functions, including
    # the above predicate
    tf.enable_eager_execution()
    # now it is true
    tf.executing_eagerly()
    x = [[2.]]
    # this will immediately start a session and do the
    # computation. The output is still a tensor, but this time with a
    # specific value.
    m = tf.matmul(x, x)
    print("hello, {}".format(m))
    a = tf.constant([[1, 2],
                 [3, 4]])
    a
    tf.matmul(a, a)
    return


def main():
    pass
