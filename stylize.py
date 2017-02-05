# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import vgg

import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib.opt.python.training import external_optimizer

from sys import stderr

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

try:
    reduce
except NameError:
    from functools import reduce


def stylize(network, initial, initial_noiseblend, content, styles, iterations,
        content_weight, content_weight_blend, style_weight, style_layer_weight_exp, style_blend_weights, tv_weight,
        learning_rate, beta1, beta2, epsilon, pooling, use_lbfgs,
        print_iterations=None, checkpoint_iterations=None):
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]
    """
    shape = (1,) + content.shape
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)
    
    layer_weight = 1.0
    STYLE_LAYERS_WEIGHTS = {}
    for style_layer in STYLE_LAYERS:
        STYLE_LAYERS_WEIGHTS[style_layer] = layer_weight
        layer_weight *= style_layer_weight_exp
    
    # normalize style layer weights
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += STYLE_LAYERS_WEIGHTS[style_layer]
    for style_layer in STYLE_LAYERS:
        STYLE_LAYERS_WEIGHTS[style_layer] /= layer_weights_sum
    
    # compute content features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net = vgg.net_preloaded(vgg_weights, image, pooling)
        content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

    # compute style features in feedforward mode
    for i in range(len(styles)):
        g = tf.Graph()
        with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
            image = tf.placeholder('float', shape=style_shapes[i])
            net = vgg.net_preloaded(vgg_weights, image, pooling)
            style_pre = np.array([vgg.preprocess(styles[i], vgg_mean_pixel)])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})
                features = np.reshape(features, (-1, features.shape[3]))
                gram = np.matmul(features.T, features) / features.size
                style_features[i][layer] = gram

    initial_content_noise_coeff = 1.0 - initial_noiseblend
                
    # make stylized image using backpropogation
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        if initial is None:
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = tf.random_normal(shape) * 0.256
        else:
            initial = np.array([vgg.preprocess(initial, vgg_mean_pixel)])
            initial = initial.astype('float32')
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = (initial) * initial_content_noise_coeff + (tf.random_normal(shape) * 0.256) * (1.0 - initial_content_noise_coeff)
        image = tf.Variable(initial)
        net = vgg.net_preloaded(vgg_weights, image, pooling)

        # content loss
        CONTENT_LAYERS_WEIGHTS = {}
        CONTENT_LAYERS_WEIGHTS['relu4_2'] = content_weight_blend
        CONTENT_LAYERS_WEIGHTS['relu5_2'] = 1.0 - content_weight_blend
        
        content_loss = 0
        content_losses = []
        for content_layer in CONTENT_LAYERS:
            content_losses.append(CONTENT_LAYERS_WEIGHTS[content_layer] * content_weight * (2 * tf.nn.l2_loss(
                    net[content_layer] - content_features[content_layer]) /
                    content_features[content_layer].size))
        content_loss += reduce(tf.add, content_losses)
        
        # style loss
        style_loss = 0
        for i in range(len(styles)):
            style_losses = []
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]
                _, height, width, number = map(lambda i: i.value, layer.get_shape())
                size = height * width * number
                feats = tf.reshape(layer, (-1, number))
                gram = tf.matmul(tf.transpose(feats), feats) / size
                style_gram = style_features[i][style_layer]
                style_losses.append(STYLE_LAYERS_WEIGHTS[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
            style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)

        # total variation denoising
        tv_y_size = _tensor_size(image[:,1:,:,:])
        tv_x_size = _tensor_size(image[:,:,1:,:])
        tv_loss = tv_weight * 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                    tv_x_size))
        # overall loss
        loss = content_loss + style_loss + tv_loss

        # optimizer setup
        def iter_callback(data=None):
            iter_callback.opt_iter += 1
            stderr.write('Iteration %4d/%4d (%f)\n' % (iter_callback.opt_iter, iter_callback.max_iter, time.time()-iter_callback.time0))
            iter_callback.time0 = time.time()
        iter_callback.opt_iter = 0
        iter_callback.max_iter = iterations
        
        if use_lbfgs == 0:
            # Sometimes Adam can benefit from the learning rate decay too, it can be visually noted when due to
            # too large rmaining learning rate, at higher iterations picture could be transformed to a worse state
            if False:
                LEARNING_RATE_INITIAL = learning_rate
                LEARNING_DECAY_STEPS = 10
                LEARNING_DECAY_BASE = 0.98
                learning_rate_decay = tf.train.exponential_decay(LEARNING_RATE_INITIAL,
                        global_step, LEARNING_DECAY_STEPS, LEARNING_DECAY_BASE,
                        staircase=True)
                train_step = tf.train.AdamOptimizer(learning_rate_decay, beta1, beta2, epsilon).minimize(loss, global_step=global_step)        
            else:
                train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)
        else:
            # setting up L-BFGS iteration parameters
            #   there are two iteration kinds: outer iteration (our loop) and inner iteration, or subiteration (the SciPy optimizer loop)
            #   ideally, we don't want to split the internal subiteration loop, as this leads to loss of precision
            #   hence in default case of L-BFGS we would want to have 1 our iteration and `iterations` subiterations
            subiterations = iterations
            if checkpoint_iterations:
                stderr.write('WARNING: checkpoint_iterations is used with L-BFGS optimizer - this will decrease the precision\n')
                stderr.write('      due to the need to split iteration seuqence to save intermediate image and show progress\n')
                if checkpoint_iterations < 50:
                    stderr.write('WARNING: checkpoint_iterations cannot be lower than 50 when using L-BFGS due to precision losses\n')
                    checkpoint_iterations = 50
                # we don't want to break up iteration sequence just for the statistics output
                if print_iterations:
                    stderr.write('WARNING: both checkpoint_iterations and print_iterations are set for L-BFGS, focring print_iterations=checkpoint_iterations\n')
                    print_iterations = checkpoint_iterations
                if subiterations > checkpoint_iterations:
                    subiterations = checkpoint_iterations
                
            elif print_iterations:
                stderr.write('WARNING: print_iterations is used with L-BFGS optimizer - this will decrease the precision\n')
                stderr.write('      due to the need to split iteration seuqence to save intermediate image and show progress\n')
                if print_iterations < 50:
                    stderr.write('WARNING: print_iterations cannot be lower than 50 when using L-BFGS due to precision losses\n')
                    print_iterations = 50
                if subiterations > print_iterations:
                    subiterations = print_iterations
            
            if subiterations != iterations and subiterations < iterations:
                # subiterations number is limited, we need to get the total amount of L-BFGS subeterations as close to `iterations` as possible
                iterations = iterations // subiterations + 1
                print_iterations = 1
                checkpoint_iterations = 1
            else:
                # subiterations number is unlimited, we only need one training iteration
                iterations = 1
            
            iter_callback.max_iter = iterations * subiterations
            
            lbfgs_optimizer = external_optimizer.ScipyOptimizerInterface(loss, callback=iter_callback, options=
                    {
                        'disp': None,
                        'maxls': 20,
                        'iprint': -1,
                        'gtol': 1e-05,
                        'eps': 1e-08,
                        'maxiter': subiterations,
                        'ftol': 2.220446049250313e-09,
                        'maxcor': 10,
                        'maxfun': 15000
                    })

        def print_progress():
            stderr.write('  content loss: %g\n' % content_loss.eval())
            stderr.write('    style loss: %g\n' % style_loss.eval())
            stderr.write('       tv loss: %g\n' % tv_loss.eval())
            stderr.write('    total loss: %g\n' % loss.eval())

        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            stderr.write('Optimization started..\n')
            if (print_iterations and print_iterations != 0):
                print_progress()
            for i in range(iterations):
                iter_callback.time0 = time.time()
                if use_lbfgs == 0:
                    train_step.run()
                    iter_callback()
                else:
                    lbfgs_optimizer.minimize(sess)                

                last_step = (i == iterations - 1)
                if last_step or (print_iterations and i % print_iterations == 0):
                    print_progress()
                
                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()
                    yield (
                        (None if last_step else i),
                        vgg.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)
                    )


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)
