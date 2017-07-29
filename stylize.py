# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.
# Copyright (c) 2017 Andrey Voroshilov

import vgg
import sqz

import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib.opt.python.training import external_optimizer

from sys import stderr
import sys

from PIL import Image

from luma_transfer import colortransfer
import common

try:
    reduce
except NameError:
    from functools import reduce

def stylize(network_file, network_type, initial, initial_noiseblend, content, styles, preserve_colors_coeff, preserve_colors_prior, iterations,
        content_weight, content_weight_blend, style_weight, style_distr_weight, style_layer_weight_exp, style_blend_weights, style_feat_type, tv_weight,
        learning_rate, beta1, beta2, epsilon, ashift, pooling, optimizer,
        print_iterations=None, checkpoint_iterations=None):
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]
    """
    activation_shift = ashift

    distribution_loss = (style_distr_weight != 0.0)
    
    shape = (1,) + content.shape
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]
    style_distr = [{} for _ in styles]

    if preserve_colors_prior == True:
        style_cnt = 0
        for i in range(len(styles)):
            styles[i] = colortransfer(content, styles[i], 'hsv')
            Image.fromarray(styles[i]).save("precolor_style%d.jpg" % (style_cnt), quality=95)
            style_cnt += 1

    if network_type == 'sqz':
        net_module = sqz
    else:
        net_module = vgg
    
    STYLE_FEATURE_TYPES_GRAM = 0
    STYLE_FEATURE_TYPES_MEAN = 1
    STYLE_FEATURE_TYPES_DISTR = 2

    style_features_type = STYLE_FEATURE_TYPES_GRAM
    if style_feat_type == 'mean':
        style_features_type = STYLE_FEATURE_TYPES_MEAN
    elif style_feat_type == 'distr':
        style_features_type = STYLE_FEATURE_TYPES_DISTR
    
    CONTENT_LAYERS = net_module.get_content_layers()
    STYLE_LAYERS = net_module.get_style_layers()
    
    vgg_weights, vgg_mean_pixel = net_module.load_net(network_file)
    if vgg_weights == None:
        print("Failed to load network\n\n")
        sys.exit(0)
    
    # calculate content layer weights
    clw = (content_weight_blend, 1.0 - content_weight_blend)
    content_layer_weight_exp = clw[1] / clw[0]
    layer_weight = 1.0
    content_layers_weights = {}
    for content_layer in CONTENT_LAYERS:
        content_layers_weights[content_layer] = layer_weight
        layer_weight *= content_layer_weight_exp

    # normalize content layer weights
    layer_weights_sum = 0
    for content_layer in CONTENT_LAYERS:
        layer_weights_sum += content_layers_weights[content_layer]
    for content_layer in CONTENT_LAYERS:
        content_layers_weights[content_layer] /= layer_weights_sum
    
    # calculate style layer weights
    layer_weight = 1.0
    style_layers_weights = {}
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = layer_weight
        layer_weight *= style_layer_weight_exp
    
    # normalize style layer weights
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] /= layer_weights_sum
    
    # compute content features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder(common.get_dtype_tf(), shape=shape)
        net = net_module.net_preloaded(vgg_weights, image, pooling)
        content_pre = np.array([net_module.preprocess(content, vgg_mean_pixel)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

    # compute style features in feedforward mode
    for i in range(len(styles)):
        g = tf.Graph()
        with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
            image = tf.placeholder(common.get_dtype_tf(), shape=style_shapes[i])
            net = net_module.net_preloaded(vgg_weights, image, pooling)
            style_pre = np.array([net_module.preprocess(styles[i], vgg_mean_pixel)])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})
                features = np.reshape(features, (-1, features.shape[3]))
                
                if distribution_loss:
                    style_distr[i][layer] = (np.mean(features, axis=0), np.std(features, axis=0))
                
                if style_features_type == STYLE_FEATURE_TYPES_GRAM:
                    # Gram matrix
                    # activation shift
                    features += np.full(features.shape, common.get_dtype_np()(activation_shift))
                    gram = np.matmul(features.T, features) / features.size
                    style_features[i][layer] = gram
                elif style_features_type == STYLE_FEATURE_TYPES_MEAN:
                    # Averaging
                    style_features[i][layer] = np.mean(features, axis=0)# / features.size
                elif style_features_type == STYLE_FEATURE_TYPES_DISTR:
                    # Full distribution parameters loss
                    style_features[i][layer] = (np.mean(features, axis=0), np.std(features, axis=0))

    initial_content_noise_coeff = 1.0 - initial_noiseblend
                
    NOISE_AMP = 64.0*0.256
    # make stylized image using backpropogation
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        if initial is None:
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = tf.random_normal(shape) * NOISE_AMP
        else:
            initial = np.array([net_module.preprocess(initial, vgg_mean_pixel)])
            initial = initial.astype(common.get_dtype_np())
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = (initial) * initial_content_noise_coeff + (tf.random_normal(shape) * NOISE_AMP) * (1.0 - initial_content_noise_coeff)

        image = tf.Variable(initial)
        net = net_module.net_preloaded(vgg_weights, image, pooling)

        # content loss
        content_loss = 0
        content_losses = []
        for content_layer in CONTENT_LAYERS:
            content_losses.append(content_layers_weights[content_layer] * content_weight * (2 * tf.nn.l2_loss(
                    net[content_layer] - content_features[content_layer]) /
                    content_features[content_layer].size))
        content_loss += reduce(tf.add, content_losses)
        
        # style loss
        style_loss = 0
        for i in range(len(styles)):
            style_losses = []
            style_distr_loss = 0
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]
                _, height, width, number = map(lambda i: i.value, layer.get_shape())
                size = height * width * number
                feats = tf.reshape(layer, (-1, number))

                if distribution_loss:
                    print("Style Layer: %s" % (style_layer))

                    EPS = 1e-5

                    style_target_distr = style_distr[i][style_layer]
                    cur_mean, cur_var = tf.nn.moments(feats, axes=[0])
                    cur_distr = (cur_mean, tf.sqrt(tf.maximum(cur_var, tf.fill([1], EPS))))
                    
                    feats_shape = feats.get_shape()
                    distr_losses = []
                    
                    feats_delta = feats - tf.add(tf.multiply(tf.div(style_target_distr[1], cur_distr[1] + tf.fill(cur_distr[1].get_shape(), EPS)), tf.subtract(feats, cur_distr[0])), style_target_distr[0])
                    style_distr_loss += style_distr_weight * tf.nn.l2_loss(feats_delta) / tf.cast(feats_shape[0] * feats_shape[1], common.get_dtype_tf())
                    
                if style_features_type == STYLE_FEATURE_TYPES_GRAM:
                    # Gram matrix
                    # activation shift
                    feats += tf.fill(feats.get_shape(), tf.cast(activation_shift, common.get_dtype_tf()))
                    gram = tf.matmul(tf.transpose(feats), feats) / size
                    style_gram = style_features[i][style_layer]
                    style_losses.append(style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
                elif style_features_type == STYLE_FEATURE_TYPES_MEAN:
                    # Averaging
                    style_target_features = style_features[i][style_layer]
                    style_current_features = tf.reduce_mean(feats, axis=0)
                    style_losses.append(style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(style_current_features - style_target_features))
                elif style_features_type == STYLE_FEATURE_TYPES_DISTR:
                    # Full distribution parameters loss
                    style_target_features = style_features[i][style_layer]
                    cur_mean, cur_var = tf.nn.moments(feats, axes=[0])
                    
                    EPS = 1e-5
                    style_current_features = (cur_mean, tf.sqrt(tf.maximum(cur_var, tf.fill([1], EPS))))
                    style_losses.append(style_layers_weights[style_layer] * 2 * (tf.nn.l2_loss(style_current_features[0] - style_target_features[0]) + tf.nn.l2_loss(style_current_features[1] - style_target_features[1])))

            style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)
            if distribution_loss:
                style_loss += style_distr_loss

        # total variation denoising
        tv_y_size = _tensor_size(image[:,1:,:,:])
        tv_x_size = _tensor_size(image[:,:,1:,:])
        tv_loss = tv_weight * 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                    tv_x_size))
        # overall loss
        loss = tf.cast(content_loss, common.get_dtype_tf()) + tf.cast(style_loss, common.get_dtype_tf()) + tf.cast(tv_loss, common.get_dtype_tf())

        # optimizer setup
        def iter_callback(data=None):
            iter_callback.opt_iter += 1
            stderr.write('Iteration %4d/%4d (%f)\n' % (iter_callback.opt_iter, iter_callback.max_iter, time.time()-iter_callback.time0))
            iter_callback.time0 = time.time()
        iter_callback.opt_iter = 0
        iter_callback.max_iter = iterations
        
        print("")
        use_scipy_optimizer = False
        if optimizer == 'adam':
            print("Using TF Adam optimizer (%f)" % (learning_rate))
            # Sometimes Adam can benefit from the learning rate decay too, it can be visually noted when due to
            # too large remaining learning rate, at higher iterations picture could be transformed to a worse state
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
            
            use_scipy_optimizer = True
            if optimizer == 'cg':
                print("Using SciPy CG optimizer")
                scipy_optimizer = external_optimizer.ScipyOptimizerInterface(loss, method='CG', callback=iter_callback, options=
                    {
                        'disp': None,
                        'gtol': 1e-05,
                        'eps': 1e-08,
                        'maxiter': subiterations,
                    })
            else:
                print("Using SciPy L-BFGS optimizer")
                scipy_optimizer = external_optimizer.ScipyOptimizerInterface(loss, method='L-BFGS-B', callback=iter_callback, options=
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

        opt_time = time.time()
            
        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            stderr.write('Optimization started..\n')
            print_progress()
            for i in range(iterations):
                iter_callback.time0 = time.time()
                if use_scipy_optimizer == 0:
                    train_step.run()
                    iter_callback()
                else:
                    scipy_optimizer.minimize(sess)                

                last_step = (i == iterations - 1)
                if last_step or (print_iterations and i % print_iterations == 0):
                    print_progress()
                
                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()
                    
                    img_out = net_module.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)
                    
                    if preserve_colors_coeff and preserve_colors_coeff != 0.0:
                        img_out_pc = colortransfer(original_image=np.clip(content, 0, 255), styled_image=np.clip(img_out, 0, 255), mode='yuv')
                        if preserve_colors_coeff == 1.0:
                            img_out = img_out_pc
                        else:
                            img_out = img_out_pc * preserve_colors_coeff + img_out * (1.0 - preserve_colors_coeff)
                        
                    yield (
                        (None if last_step else i),
                        img_out
                    )

        print("Optimization time: %fs" % (time.time() - opt_time))
                    
def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)
