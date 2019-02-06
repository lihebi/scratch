#!/usr/bin/env python3

"""
Source is models/research/gan/tutorial.ipynb

More sources:
- https://github.com/eriklindernoren/Keras-GAN
- https://github.com/eriklindernoren/PyTorch-GAN
- https://github.com/pytorch/examples/tree/master/dcgan
- https://github.com/wiseodd/generative-models

"""


import sys

sys.path.append('/home/hebi/github/reading/tensorflow-models/research/gan')
sys.path.append('/home/hebi/github/reading/tensorflow-models/research/')
sys.path.append('/home/hebi/github/reading/tensorflow-models/research/slim')

import matplotlib.pyplot as plt
import numpy as np


import tensorflow as tf
tfgan = tf.contrib.gan

# TFGAN MNIST examples from `tensorflow/models`.
from mnist import data_provider
from mnist import util

# TF-Slim data provider.
from datasets import download_and_convert_mnist

queues = tf.contrib.slim.queues
layers = tf.contrib.layers
ds = tf.contrib.distributions
framework = tf.contrib.framework

leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

def visualize_digits(tensor_to_visualize):
    """Visualize an image once. Used to visualize generator before training.
    
    Args:
        tensor_to_visualize: An image tensor to visualize. A python Tensor.
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with tf.contrib.slim.queues.QueueRunners(sess):
            images_np = sess.run(tensor_to_visualize)
    plt.axis('off')
    plt.imshow(np.squeeze(images_np), cmap='gray')
    
def prepare_data():
    MNIST_DATA_DIR = '/tmp/mnist-data'
    if not tf.gfile.Exists(MNIST_DATA_DIR):
        tf.gfile.MakeDirs(MNIST_DATA_DIR)
    download_and_convert_mnist.run(MNIST_DATA_DIR)

    # FIXME why reset graph here?
    tf.reset_default_graph()
    # Define our input pipeline. Pin it to the CPU so that the GPU can be reserved
    # for forward and backwards propogation.
    batch_size = 32
    with tf.device('/cpu:0'):
        # data provider FIXME why returning three fields?
        real_images, _, _ = data_provider.provide_data(
            'train', batch_size, MNIST_DATA_DIR)

    # Sanity check that we're getting images.
    check_real_digits = tfgan.eval.image_reshaper(
        real_images[:20,...], num_cols=10)
    visualize_digits(check_real_digits)
    # FIXME is this a tensor or an iterator?
    return real_images

def generator_fn(noise, weight_decay=2.5e-5, is_training=True):
    with framework.arg_scope(
        [layers.fully_connected, layers.conv2d_transpose],
        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
        weights_regularizer=layers.l2_regularizer(weight_decay)),\
    framework.arg_scope([layers.batch_norm], is_training=is_training,
                        zero_debias_moving_mean=True):
        net = layers.fully_connected(noise, 1024)
        net = layers.fully_connected(net, 7 * 7 * 256)
        net = tf.reshape(net, [-1, 7, 7, 256])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
        # Make sure that generator output is in the same range as `inputs`
        # ie [-1, 1].
        net = layers.conv2d(net, 1, 4, normalizer_fn=None, activation_fn=tf.tanh)

        return net
    
def discriminator_fn(img, unused_conditioning, weight_decay=2.5e-5,
                     is_training=True):
    with framework.arg_scope(
        [layers.conv2d, layers.fully_connected],
        activation_fn=leaky_relu, normalizer_fn=None,
        weights_regularizer=layers.l2_regularizer(weight_decay),
        biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        with framework.arg_scope([layers.batch_norm], is_training=is_training):
            net = layers.fully_connected(net, 1024, normalizer_fn=layers.batch_norm)
        return layers.linear(net, 1)

def gan_loss():
    # We can use the minimax loss from the original paper.
    vanilla_gan_loss = tfgan.gan_loss(
        gan_model,
        generator_loss_fn=tfgan.losses.minimax_generator_loss,
        discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss)
    # We can use the Wasserstein loss (https://arxiv.org/abs/1701.07875) with the 
    # gradient penalty from the improved Wasserstein loss paper 
    # (https://arxiv.org/abs/1704.00028).
    improved_wgan_loss = tfgan.gan_loss(
        gan_model,
        # We make the loss explicit for demonstration, even though the default is 
        # Wasserstein loss.
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        gradient_penalty_weight=1.0)
    # We can also define custom losses to use with the rest of the TFGAN framework.
    def silly_custom_generator_loss(gan_model, add_summaries=False):
        return tf.reduce_mean(gan_model.discriminator_gen_outputs)
    def silly_custom_discriminator_loss(gan_model, add_summaries=False):
        return (tf.reduce_mean(gan_model.discriminator_gen_outputs) -
                tf.reduce_mean(gan_model.discriminator_real_outputs))
    custom_gan_loss = tfgan.gan_loss(
        gan_model,
        generator_loss_fn=silly_custom_generator_loss,
        discriminator_loss_fn=silly_custom_discriminator_loss)
    # Sanity check that we can evaluate our losses.
    for gan_loss, name in [(vanilla_gan_loss, 'vanilla loss'), 
                           (improved_wgan_loss, 'improved wgan loss'), 
                           (custom_gan_loss, 'custom loss')]:
        evaluate_tfgan_loss(gan_loss, name)
    return vanilla_gan_loss

def gan_model():
    noise_dims = 64
    real_images = prepare_data()
    gan_model = tfgan.gan_model(
        generator_fn,
        discriminator_fn,
        real_data=real_images,
        generator_inputs=tf.random_normal([batch_size, noise_dims]))
    # Sanity check that generated images before training are garbage.
    check_generated_digits = tfgan.eval.image_reshaper(
        gan_model.generated_data[:20,...], num_cols=10)
    visualize_digits(check_generated_digits)
    # loss
    loss = gan_loss()
    # train op
    generator_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
    discriminator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5)
    gan_train_ops = tfgan.gan_train_ops(
        gan_model,
        improved_wgan_loss,
        generator_optimizer,
        discriminator_optimizer)
    # training. The original tutorial includes evaluation metrics
    # construction, and a detailed step control. However, here I'm
    # using gan_train function instead.
    #
    # Run the train ops in the alternating training scheme.
    tfgan.gan_train(
        gan_train_ops,
        hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps)],
        logdir=FLAGS.train_log_dir)

def gan_estimator():
    """GAN using estimator APIs
    """
    tf.reset_default_graph()

    def _get_train_input_fn(batch_size, noise_dims):
        def train_input_fn():
            with tf.device('/cpu:0'):
                real_images, _, _ = data_provider.provide_data(
                    'train', batch_size, MNIST_DATA_DIR)
            noise = tf.random_normal([batch_size, noise_dims])
            return noise, real_images
        return train_input_fn


    def _get_predict_input_fn(batch_size, noise_dims):
        def predict_input_fn():
            noise = tf.random_normal([batch_size, noise_dims])
            return noise
        return predict_input_fn
    BATCH_SIZE = 32
    NOISE_DIMS = 64
    NUM_STEPS = 2000

    # Initialize GANEstimator with options and hyperparameters.
    gan_estimator = tfgan.estimator.GANEstimator(
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        generator_optimizer=tf.train.AdamOptimizer(0.001, 0.5),
        discriminator_optimizer=tf.train.AdamOptimizer(0.0001, 0.5),
        add_summaries=tfgan.estimator.SummaryType.IMAGES)

    # Train estimator.
    train_input_fn = _get_train_input_fn(BATCH_SIZE, NOISE_DIMS)
    start_time = time.time()
    # (HEBI: Train)
    gan_estimator.train(train_input_fn, max_steps=NUM_STEPS)
    time_since_start = (time.time() - start_time) / 60.0
    print('Time since start: %f m' % time_since_start)
    print('Steps per min: %f' % (NUM_STEPS / time_since_start))

    # Now, visualize some examples, i.e. do the inference
    def _get_next(iterable):
        try:
            return iterable.next()  # Python 2.x.x
        except AttributeError:
            return iterable.__next__()  # Python 3.x.x

    # Run inference.
    predict_input_fn = _get_predict_input_fn(36, NOISE_DIMS)
    prediction_iterable = gan_estimator.predict(
        predict_input_fn, hooks=[tf.train.StopAtStepHook(last_step=1)])
    predictions = [_get_next(prediction_iterable) for _ in xrange(36)]

    try: # Close the predict session.
        _get_next(prediction_iterable)
    except StopIteration:
        pass

    # Nicely tile output and visualize.
    image_rows = [np.concatenate(predictions[i:i+6], axis=0) for i in
                  range(0, 36, 6)]
    tiled_images = np.concatenate(image_rows, axis=1)

    # Visualize.
    plt.axis('off')
    plt.imshow(np.squeeze(tiled_images), cmap='gray')

    
def conditional_generator_fn(inputs, weight_decay=2.5e-5, is_training=True):
    noise, one_hot_labels = inputs
  
    with framework.arg_scope(
        [layers.fully_connected, layers.conv2d_transpose],
        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
        weights_regularizer=layers.l2_regularizer(weight_decay)),\
    framework.arg_scope([layers.batch_norm], is_training=is_training,
                        zero_debias_moving_mean=True):
        net = layers.fully_connected(noise, 1024)
        net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)
        net = layers.fully_connected(net, 7 * 7 * 128)
        net = tf.reshape(net, [-1, 7, 7, 128])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
        # Make sure that generator output is in the same range as `inputs`
        # ie [-1, 1].
        net = layers.conv2d(net, 1, 4, normalizer_fn=None, activation_fn=tf.tanh)

        return net
def conditional_discriminator_fn(img, conditioning, weight_decay=2.5e-5):
    _, one_hot_labels = conditioning
    with framework.arg_scope(
        [layers.conv2d, layers.fully_connected],
        activation_fn=leaky_relu, normalizer_fn=None,
        weights_regularizer=layers.l2_regularizer(weight_decay),
        biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        net = layers.fully_connected(net, 1024, normalizer_fn=layers.batch_norm)
        
        return layers.linear(net, 1)
def conditional():   
    noise_dims = 64
    # FIXME (HEBI: one hot label creation)
    real_images = prepare_data()
    conditional_gan_model = tfgan.gan_model(
        # (HEBI: using the conditional generator and discriminator)
        generator_fn=conditional_generator_fn,
        discriminator_fn=conditional_discriminator_fn,
        real_data=real_images,
        # (HEBI: the one hot label is used as input)
        generator_inputs=(tf.random_normal([batch_size, noise_dims]), 
                          one_hot_labels))

    # Sanity check that currently generated images are garbage.
    cond_generated_data_to_visualize = tfgan.eval.image_reshaper(
        conditional_gan_model.generated_data[:20,...], num_cols=10)
    visualize_digits(cond_generated_data_to_visualize)
    # Loss
    gan_loss = tfgan.gan_loss(
        conditional_gan_model, gradient_penalty_weight=1.0)
    # Sanity check that we can evaluate our losses.
    evaluate_tfgan_loss(gan_loss)
    # train ops
    generator_optimizer = tf.train.AdamOptimizer(0.0009, beta1=0.5)
    discriminator_optimizer = tf.train.AdamOptimizer(0.00009, beta1=0.5)
    gan_train_ops = tfgan.gan_train_ops(
        conditional_gan_model,
        gan_loss,
        generator_optimizer,
        discriminator_optimizer)
    # Run the train ops in the alternating training scheme.
    tfgan.gan_train(
        gan_train_ops,
        hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps)],
        logdir=FLAGS.train_log_dir)
    

def infogan_generator(inputs, categorical_dim, weight_decay=2.5e-5,
                      is_training=True):
    unstructured_noise, cat_noise, cont_noise = inputs
    cat_noise_onehot = tf.one_hot(cat_noise, categorical_dim)
    all_noise = tf.concat([unstructured_noise, cat_noise_onehot, cont_noise], axis=1)
    
    with framework.arg_scope(
        [layers.fully_connected, layers.conv2d_transpose],
        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
        weights_regularizer=layers.l2_regularizer(weight_decay)),\
    framework.arg_scope([layers.batch_norm], is_training=is_training):
        net = layers.fully_connected(all_noise, 1024)
        net = layers.fully_connected(net, 7 * 7 * 128)
        net = tf.reshape(net, [-1, 7, 7, 128])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
        # Make sure that generator output is in the same range as `inputs`
        # ie [-1, 1].
        net = layers.conv2d(net, 1, 4, normalizer_fn=None, activation_fn=tf.tanh)
    
        return net

def infogan_discriminator(img, unused_conditioning, weight_decay=2.5e-5,
                          categorical_dim=10, continuous_dim=2, is_training=True):
    with framework.arg_scope(
        [layers.conv2d, layers.fully_connected],
        activation_fn=leaky_relu, normalizer_fn=None,
        weights_regularizer=layers.l2_regularizer(weight_decay),
        biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)
    
        logits_real = layers.fully_connected(net, 1, activation_fn=None)

        # Recognition network for latent variables has an additional layer
        with framework.arg_scope([layers.batch_norm], is_training=is_training):
            encoder = layers.fully_connected(
                net, 128, normalizer_fn=layers.batch_norm)

        # Compute logits for each category of categorical latent.
        logits_cat = layers.fully_connected(
            encoder, categorical_dim, activation_fn=None)
        q_cat = ds.Categorical(logits_cat)

        # Compute mean for Gaussian posterior of continuous latents.
        mu_cont = layers.fully_connected(
            encoder, continuous_dim, activation_fn=None)
        sigma_cont = tf.ones_like(mu_cont)
        q_cont = ds.Normal(loc=mu_cont, scale=sigma_cont)

        return logits_real, [q_cat, q_cont]

def infogan():
    real_images = prepare_data()
    # Dimensions of the structured and unstructured noise dimensions.
    cat_dim, cont_dim, noise_dims = 10, 2, 64

    # (HEBI: using infogan generator)
    generator_fn = functools.partial(infogan_generator, categorical_dim=cat_dim)
    discriminator_fn = functools.partial(
        infogan_discriminator, categorical_dim=cat_dim,
        continuous_dim=cont_dim)
    unstructured_inputs, structured_inputs = util.get_infogan_noise(
        batch_size, cat_dim, cont_dim, noise_dims)

    infogan_model = tfgan.infogan_model(
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        real_data=real_images,
        unstructured_generator_inputs=unstructured_inputs,
        structured_generator_inputs=structured_inputs)
    infogan_loss = tfgan.gan_loss(
        infogan_model,
        gradient_penalty_weight=1.0,
        # (HEBI: the mutual information penalty!!)
        mutual_information_penalty_weight=1.0)

    # Sanity check that we can evaluate our losses.
    evaluate_tfgan_loss(infogan_loss)
    # train ops
    generator_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
    discriminator_optimizer = tf.train.AdamOptimizer(0.00009, beta1=0.5)
    gan_train_ops = tfgan.gan_train_ops(
        infogan_model,
        infogan_loss,
        generator_optimizer,
        discriminator_optimizer)
    # train
    tfgan.gan_train(
        gan_train_ops,
        hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps)],
        logdir=FLAGS.train_log_dir)

def load_from_checkpoint():
    """FIXME this is not runnable.
    """
    # ADAM variables are causing the checkpoint reload to choke, so omit them when 
    # doing variable remapping.
    var_dict = {x.op.name: x for x in 
                tf.contrib.framework.get_variables('Generator/') 
                if 'Adam' not in x.name}
    tf.contrib.framework.init_from_checkpoint(
        './mnist/data/infogan_model.ckpt', var_dict)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        display_img_np = sess.run(display_img)
    plt.axis('off')
    plt.imshow(np.squeeze(display_img_np), cmap='gray')
    plt.show()


    
##############################
# Examples from tensorflow/contrib/gan/README.md
##############################

def unconditional():
    # Set up the input.
    images = mnist_data_provider.provide_data(FLAGS.batch_size)
    noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dims])

    # Build the generator and discriminator.
    gan_model = tfgan.gan_model(
        generator_fn=mnist.unconditional_generator,  # you define
        discriminator_fn=mnist.unconditional_discriminator,  # you define
        real_data=images,
        generator_inputs=noise)

    # Build the GAN loss.
    gan_loss = tfgan.gan_loss(
        gan_model,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss)

    # Create the train ops, which calculate gradients and apply updates to weights.
    train_ops = tfgan.gan_train_ops(
        gan_model,
        gan_loss,
        generator_optimizer=tf.train.AdamOptimizer(gen_lr, 0.5),
        discriminator_optimizer=tf.train.AdamOptimizer(dis_lr, 0.5))

    # Run the train ops in the alternating training scheme.
    tfgan.gan_train(
        train_ops,
        hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps)],
        logdir=FLAGS.train_log_dir)
def conditional():
    # Set up the input.
    images, one_hot_labels = mnist_data_provider.provide_data(FLAGS.batch_size)
    noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dims])

    # Build the generator and discriminator.
    gan_model = tfgan.gan_model(
        generator_fn=mnist.conditional_generator,  # you define
        discriminator_fn=mnist.conditional_discriminator,  # you define
        real_data=images,
        generator_inputs=(noise, one_hot_labels))

    # The rest is the same as in the unconditional case.
    # ...
def adverserial_loss():
    # Set up the input pipeline.
    images = image_provider.provide_data(FLAGS.batch_size)

    # Build the generator and discriminator.
    gan_model = tfgan.gan_model(
        generator_fn=nets.autoencoder,  # you define
        discriminator_fn=nets.discriminator,  # you define
        real_data=images,
        generator_inputs=images)

    # Build the GAN loss and standard pixel loss.
    gan_loss = tfgan.gan_loss(
        gan_model,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        gradient_penalty=1.0)
    l1_pixel_loss = tf.norm(gan_model.real_data - gan_model.generated_data, ord=1)

    # Modify the loss tuple to include the pixel loss.
    gan_loss = tfgan.losses.combine_adversarial_loss(
        gan_loss, gan_model, l1_pixel_loss, weight_factor=FLAGS.weight_factor)

    # The rest is the same as in the unconditional case.
    # ...
    
def image2image():
    # Set up the input pipeline.
    input_image, target_image = data_provider.provide_data(FLAGS.batch_size)

    # Build the generator and discriminator.
    gan_model = tfgan.gan_model(
        generator_fn=nets.generator,  # you define
        discriminator_fn=nets.discriminator,  # you define
        real_data=target_image,
        generator_inputs=input_image)

    # Build the GAN loss and standard pixel loss.
    gan_loss = tfgan.gan_loss(
        gan_model,
        generator_loss_fn=tfgan.losses.least_squares_generator_loss,
        discriminator_loss_fn=tfgan.losses.least_squares_discriminator_loss)
    l1_pixel_loss = tf.norm(gan_model.real_data - gan_model.generated_data, ord=1)

    # Modify the loss tuple to include the pixel loss.
    gan_loss = tfgan.losses.combine_adversarial_loss(
        gan_loss, gan_model, l1_pixel_loss, weight_factor=FLAGS.weight_factor)

    # The rest is the same as in the unconditional case.
    # ...
    
def infogan():
    # Set up the input pipeline.
    images = mnist_data_provider.provide_data(FLAGS.batch_size)

    # Build the generator and discriminator.
    gan_model = tfgan.infogan_model(
        generator_fn=mnist.infogan_generator,  # you define
        discriminator_fn=mnist.infogran_discriminator,  # you define
        real_data=images,
        unstructured_generator_inputs=unstructured_inputs,  # you define
        structured_generator_inputs=structured_inputs)  # you define

    # Build the GAN loss with mutual information penalty.
    gan_loss = tfgan.gan_loss(
        gan_model,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        gradient_penalty=1.0,
        mutual_information_penalty_weight=1.0)

    # The rest is the same as in the unconditional case.
    # ...
    
def custom_model():
    # Set up the input pipeline.
    images = mnist_data_provider.provide_data(FLAGS.batch_size)
    noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dims])

    # Manually build the generator and discriminator.
    with tf.variable_scope('Generator') as gen_scope:
      generated_images = generator_fn(noise)
    with tf.variable_scope('Discriminator') as dis_scope:
      discriminator_gen_outputs = discriminator_fn(generated_images)
    with variable_scope.variable_scope(dis_scope, reuse=True):
      discriminator_real_outputs = discriminator_fn(images)
    generator_variables = variables_lib.get_trainable_variables(gen_scope)
    discriminator_variables = variables_lib.get_trainable_variables(dis_scope)
    # Depending on what TFGAN features you use, you don't always need to supply
    # every `GANModel` field. At a minimum, you need to include the discriminator
    # outputs and variables if you want to use TFGAN to construct losses.
    gan_model = tfgan.GANModel(
        generator_inputs,
        generated_data,
        generator_variables,
        gen_scope,
        generator_fn,
        real_data,
        discriminator_real_outputs,
        discriminator_gen_outputs,
        discriminator_variables,
        dis_scope,
        discriminator_fn)

    # The rest is the same as the unconditional case.
    # ...
    
