# Copyright (c) 2017 Andrey Voroshilov

import os
import tensorflow as tf
import numpy as np
import scipy.io
import common

# SqueezeNet v1.1 (signature pool 1/3/5)

# Classifier not included
#   this list of layers is not used in the code, and present just for convenience
SQUEEZENET_LAYERS = (
    'conv1',
    'pool1',
    'fire2/squeeze1x1',
    'fire2/expand1x1',
    'fire2/expand3x3',
    'fire2/concat',
    'fire3/squeeze1x1',
    'fire3/expand1x1',
    'fire3/expand3x3',
    'fire3/concat',
    'pool3',
    'fire4/squeeze1x1',
    'fire4/expand1x1',
    'fire4/expand3x3',
    'fire4/concat',
    'fire5/squeeze1x1',
    'fire5/expand1x1',
    'fire5/expand3x3',
    'fire5/concat',
    'pool5',
    'fire6/squeeze1x1',
    'fire6/expand1x1',
    'fire6/expand3x3',
    'fire6/concat',
    'fire7/squeeze1x1',
    'fire7/expand1x1',
    'fire7/expand3x3',
    'fire7/concat',
    'fire8/squeeze1x1',
    'fire8/expand1x1',
    'fire8/expand3x3',
    'fire8/concat',
    'fire9/squeeze1x1'
    'fire9/expand1x1',
    'fire9/expand3x3',
    'fire9/concat',
)

CONTENT_LAYERS = (
    'conv1_actv',
    'fire2/squeeze1x1_actv',
    'fire3/squeeze1x1_actv',
    'fire4/squeeze1x1_actv',
    'fire5/squeeze1x1_actv',
    'fire6/squeeze1x1_actv',
    'fire7/squeeze1x1_actv',
    'fire8/squeeze1x1_actv',
    'fire9/squeeze1x1_actv'
    )

STYLE_POSTFIX1 = '_actv'
STYLE_LAYERS = (
    'fire2/expand1x1' + STYLE_POSTFIX1,
    'fire2/expand3x3' + STYLE_POSTFIX1,
    'fire3/expand1x1' + STYLE_POSTFIX1,
    'fire3/expand3x3' + STYLE_POSTFIX1,
    'fire4/expand1x1' + STYLE_POSTFIX1,
    'fire4/expand3x3' + STYLE_POSTFIX1,
    'fire5/expand1x1' + STYLE_POSTFIX1,
    'fire5/expand3x3' + STYLE_POSTFIX1,
    'fire6/expand1x1' + STYLE_POSTFIX1,
    'fire6/expand3x3' + STYLE_POSTFIX1,
    'fire7/expand1x1' + STYLE_POSTFIX1,
    'fire7/expand3x3' + STYLE_POSTFIX1,
    'fire8/expand1x1' + STYLE_POSTFIX1,
    'fire8/expand3x3' + STYLE_POSTFIX1,
    'fire9/expand1x1' + STYLE_POSTFIX1,
    'fire9/expand3x3' + STYLE_POSTFIX1
    )

def load_net(data_path):
    if data_path is None:
        data_path = 'sqz_notop.mat'
    if not os.path.isfile(data_path):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % data_path)

    weights_raw = scipy.io.loadmat(data_path)
    
    # Converting to needed type
    weights = {}
    for name in weights_raw:
        weights[name] = []
        # skipping '__version__', '__header__', '__globals__'
        if name[0:2] != '__':
            kernels, bias = weights_raw[name][0]
            weights[name].append( kernels.astype(common.get_dtype_np()) )
            weights[name].append( bias.astype(common.get_dtype_np()) )
    
    mean_pixel = np.array([104.006, 116.669, 122.679], dtype=common.get_dtype_np())
    return weights, mean_pixel

def preprocess(image, mean_pixel):
    swap_img = np.array(image)
    image[:, :, 0] = swap_img[:, :, 2]
    image[:, :, 2] = swap_img[:, :, 0]
    return image - mean_pixel

def unprocess(image, mean_pixel):
    swap_img = np.array(image)
    image[:, :, 0] = swap_img[:, :, 2]
    image[:, :, 2] = swap_img[:, :, 0]
    return image + mean_pixel

def get_weights_biases(preloaded, layer_name):
    weights, biases = preloaded[layer_name]
    biases = biases.reshape(-1)
    return (weights, biases)

def fire_cluster(net, x, preloaded, cluster_name):
    # central - squeeze
    layer_name = cluster_name + '/squeeze1x1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID')
    x = _act_layer(net, layer_name + '_actv', x)
    
    # left - expand 1x1
    layer_name = cluster_name + '/expand1x1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x_l = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID')
    x_l = _act_layer(net, layer_name + '_actv', x_l)

    # right - expand 3x3
    layer_name = cluster_name + '/expand3x3'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x_r = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='SAME')
    x_r = _act_layer(net, layer_name + '_actv', x_r)
    
    # concatenate expand 1x1 (left) and expand 3x3 (right)
    x = tf.concat([x_l, x_r], 3)
    net[cluster_name + '/concat_conc'] = x
    
    return x

def net_preloaded(preloaded, input_image, pooling, needs_classifier=False):
    net = {}

    x = tf.cast(input_image, common.get_dtype_tf())

    # Feature extractor
    #####################
    
    bypass = False
    
    # conv1 cluster
    layer_name = 'conv1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID', stride=(2, 2))
    x = _act_layer(net, layer_name + '_actv', x)
    x = _pool_layer(net, 'pool1_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

    # fire2 + fire3 clusters
    x = fire_cluster(net, x, preloaded, cluster_name='fire2')
    fire2_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire3')
    if bypass == True:
        x = x + fire2_bypass
    x = _pool_layer(net, 'pool3_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

    # fire4 + fire5 clusters
    x = fire_cluster(net, x, preloaded, cluster_name='fire4')
    fire4_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire5')
    if bypass == True:
        x = x + fire4_bypass
    x = _pool_layer(net, 'pool5_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

    # remainder (no pooling)
    x = fire_cluster(net, x, preloaded, cluster_name='fire6')
    fire6_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire7')
    if bypass == True:
        x = x + fire6_bypass
    x = fire_cluster(net, x, preloaded, cluster_name='fire8')
    x = fire_cluster(net, x, preloaded, cluster_name='fire9')
    
    # Classifier
    #####################
    if needs_classifier == True:
        # Fixed global avg pool/softmax classifier:
        # [227, 227, 3] -> 1000 classes
        layer_name = 'conv10'
        weights, biases = get_weights_biases(preloaded, layer_name)
        x = _conv_layer(net, layer_name + '_conv', x, weights, biases)
        x = _act_layer(net, layer_name + '_actv', x)
        
        # Global Average Pooling
        x = tf.nn.avg_pool(x, ksize=(1, 13, 13, 1), strides=(1, 1, 1, 1), padding='VALID')
        net['classifier_pool'] = x
        
        x = tf.nn.softmax(x)
        net['classifier_actv'] = x
    
    return net
    
def _conv_layer(net, name, input, weights, bias, padding='SAME', stride=(1, 1)):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, stride[0], stride[1], 1),
            padding=padding)
    x = tf.nn.bias_add(conv, bias)
    net[name] = x
    return x

def _act_layer(net, name, input):
    #x = tf.nn.relu(input)
    x = tf.nn.relu(input)
    net[name] = x
    return x
    
def _pool_layer(net, name, input, pooling, size=(2, 2), stride=(3, 3), padding='SAME'):
    if pooling == 'avg':
        x = tf.nn.avg_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                padding=padding)
    else:
        x = tf.nn.max_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                padding=padding)
    net[name] = x
    return x
   
def get_content_layers():
    return CONTENT_LAYERS

def get_style_layers():
    return STYLE_LAYERS    
    