# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.
# Copyright (c) 2017 Andrey Voroshilov

import os

import numpy as np
import scipy.misc

from stylize import stylize
import common
import common_images as comimg

import build_collage

import time
import math
from argparse import ArgumentParser

# default arguments
CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
STYLE_DISTR_WEIGHT = 0
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 1000
POOLING = 'max'
OPTIMIZER = 'lbfgs'
CHECKPOINT_OUTPUT = 'checkpoint%s.jpg'
MAX_HIERARCHY = 1
INITIAL_NOISEBLEND = 0.0
ACTIVATION_SHIFT = 0.0
PRESERVE_COLORS = 'none'
NETWORK_TYPE = 'vgg'
STYLE_FEATURE_TYPE = 'gram'

def build_parser():
    ps = ArgumentParser()
    ps.add_argument('--content',            dest='content', help='content image', metavar='CONTENT', required=True)
    ps.add_argument('--styles',             dest='styles', help='one or more style images', nargs='+', metavar='STYLE', required=True)
    ps.add_argument('--output',             dest='output', help='output path', metavar='OUTPUT')
    ps.add_argument('--iterations',         dest='iterations', type=int, help='iterations (default %(default)s)', metavar='ITERATIONS', default=ITERATIONS)
    ps.add_argument('--print-iterations',   dest='print_iterations', type=int, help='statistics printing frequency', metavar='PRINT_ITERATIONS')
    ps.add_argument('--checkpoint-output',  dest='checkpoint_output', help='checkpoint output format, e.g. output%%s.jpg', metavar='OUTPUT', default=CHECKPOINT_OUTPUT)
    ps.add_argument('--checkpoint-iterations', dest='checkpoint_iterations', type=int, help='checkpoint frequency', metavar='CHECKPOINT_ITERATIONS')
    ps.add_argument('--width',              dest='width', type=int, help='output width', metavar='WIDTH')
    ps.add_argument('--style-scales',       dest='style_scales', type=float, nargs='+', help='one or more style scales', metavar='STYLE_SCALE')
    ps.add_argument('--network-file',       dest='network_file', help='path to pretrained network parameters', metavar='NETWORK_FILE')
    ps.add_argument('--network-type',       dest='network_type', help='neural network model to use: vgg / sqz (default %(default)s)', metavar='NETWORK_TYPE', default=NETWORK_TYPE)
    ps.add_argument('--content-weight-blend', dest='content_weight_blend', type=float, metavar='CONTENT_WEIGHT_BLEND', default=CONTENT_WEIGHT_BLEND,
                                            help='content weight blend, conv4_2 * blend + conv5_2 * (1-blend) (default %(default)s)')
    ps.add_argument('--content-weight',     dest='content_weight', type=float, help='content weight (default %(default)s)', metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    ps.add_argument('--style-weight',       dest='style_weight', type=float, help='style weight (default %(default)s)', metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    ps.add_argument('--style-distr-weight', dest='style_distr_weight', type=float, help='style distribution weight (default %(default)s)', metavar='STYLE_DISTR_WEIGHT', default=STYLE_DISTR_WEIGHT)
    ps.add_argument('--style-layer-weight-exp', dest='style_layer_weight_exp', type=float, metavar='STYLE_LAYER_WEIGHT_EXP', default=STYLE_LAYER_WEIGHT_EXP,
                                            help='style layer weight exponentional increase - weight(layer<n+1>) = weight_exp*weight(layer<n>) (default %(default)s)')
    ps.add_argument('--style-blend-weights', dest='style_blend_weights', type=float, help='style blending weights', nargs='+', metavar='STYLE_BLEND_WEIGHT')
    ps.add_argument('--style-feat-type',    dest='style_feat_type', help='style feature type, \'gram\' or \'mean\' (default %(default)s)', metavar='STYLE_FEATURE_TYPE', default=STYLE_FEATURE_TYPE)
    ps.add_argument('--tv-weight',          dest='tv_weight', type=float, help='total variation regularization weight (default %(default)s)', metavar='TV_WEIGHT', default=TV_WEIGHT)
    ps.add_argument('--initial',            dest='initial', help='initial image', metavar='INITIAL')
    ps.add_argument('--initial-noiseblend', dest='initial_noiseblend', type=float, metavar='INITIAL_NOISEBLEND', default=INITIAL_NOISEBLEND,
                                            help='ratio of blending initial image with normalized noise (if no initial image specified, content image is used) (default %(default)s)')
    ps.add_argument('--preserve-colors',    dest='preserve_colors', help='preserve colors of original content image, values: none/before/all/out/interm (default %(default)s)', metavar='PRESERVE_COLORS', default=PRESERVE_COLORS)
    ps.add_argument('--pooling',            dest='pooling', help='pooling layer configuration: max or avg (default %(default)s)', metavar='POOLING', default=POOLING)
    ps.add_argument('--optim',              dest='optimizer', help='optimizer to minimize the loss: adam, lbfgs or cg (default %(default)s)', metavar='OPTIMIZER', default=OPTIMIZER)
    ps.add_argument('--max-hierarchy',      dest='max_hierarchy', type=int, metavar='MAX_HIERARCHY', default=MAX_HIERARCHY,
                                            help='maximum amount of downscaling steps to produce initial guess for the final step (default %(default)s)')
    ps.add_argument('--h-preserve-colors',  dest='h_preserve_colors', action='store_true', help='preserving colors for intermediate tiles for hierarchical style trasnfer (output colors are controlled by different key)')
    ps.add_argument('--ashift',             dest='ashift', type=float, metavar='ACTIVATION_SHIFT', default=ACTIVATION_SHIFT,
                                            help='activation shift: Gram matrix is now (F+ashift)(F+ashift)^T (default %(default)s - matches old behavior)')
    ps.add_argument('--out-postfix',        dest='out_postfix', help='when the name is auto-generated, add custom postfix', metavar='OUT_POSTFIX')
    ps.add_argument('--no-collage',         dest='no_collage', action='store_true', help='do not append downscaled content and style to the result')

    # Adam specific arguments
    ps.add_argument('--learning-rate',      dest='learning_rate', type=float, help='learning rate (default %(default)s)', metavar='LEARNING_RATE', default=LEARNING_RATE)
    ps.add_argument('--beta1',              dest='beta1', type=float, help='Adam: beta1 parameter (default %(default)s)', metavar='BETA1', default=BETA1)
    ps.add_argument('--beta2',              dest='beta2', type=float, help='Adam: beta2 parameter (default %(default)s)', metavar='BETA2', default=BETA2)
    ps.add_argument('--eps',                dest='epsilon', type=float, help='Adam: epsilon parameter (default %(default)s)', metavar='EPSILON', default=EPSILON)

    
    
    return ps

def main():
    parser = build_parser()
    options = parser.parse_args()

    content_basepath, content_filename = os.path.split(options.content)

    style_basepaths = []
    style_filenames = []
    for style in options.styles:
        basepath, filename = os.path.split(style)
        style_basepaths.append(basepath)
        style_filenames.append(filename)

    if options.output is None:
        # For now, only works with the first style
        style_filename = style_filenames[0]
        
        out_stylewe = int(options.style_layer_weight_exp * 10)
        out_ashift = int(options.ashift)
        out_contentwe = int(options.content_weight_blend * 10)
        
        postfix = ""
        if options.out_postfix is not None:
            postfix = "_" + options.out_postfix

        out_sft = ""
        if options.style_feat_type != 'gram':
            out_sft = "_" + options.style_feat_type
        
        out_preserve = ""
        if options.preserve_colors != 'none':
            if options.preserve_colors == 'before':
                out_preserve = "_bpc"
            else:
                out_preserve = "_pc"
        
        out_distr_weight = ""
        if options.style_distr_weight != 0.0:
            out_distr_weight = "_sdw%03d" % (int(options.style_distr_weight))
        
        
        options.output = "t_%s_%s_%s%04d_h%d_p%s_sw%05d%s_swe%02d_cwe%02d_as%03d_%s%s%s%s.jpg" % (content_filename, style_filename, options.optimizer, options.iterations, options.max_hierarchy, options.pooling, int(options.style_weight), out_distr_weight, out_stylewe, out_contentwe, out_ashift, options.network_type, out_sft, out_preserve, postfix)
        
        print("Using auto-generated output filename: %s" % (options.output))

    content_image = comimg.imread(options.content).astype(common.get_dtype_np())
    style_images = [comimg.imread(style).astype(common.get_dtype_np()) for style in options.styles]

    width = options.width
    if width is not None:
        new_shape = (int(math.floor(float(content_image.shape[0]) /
                content_image.shape[1] * width)), width)
        content_image = scipy.misc.imresize(content_image, new_shape)
        
    # TODO: remove this probably, since double doswnscale could affect quality
    #   however, it could save some time if the style image is a lot bigger than content
    target_shape = content_image.shape
    for i in range(len(style_images)):
        style_scale = STYLE_SCALE
        if options.style_scales is not None:
            style_scale = options.style_scales[i]
        style_images[i] = scipy.misc.imresize(style_images[i], style_scale *
                target_shape[1] / style_images[i].shape[1])

    style_blend_weights = options.style_blend_weights
    if style_blend_weights is None:
        # default is equal weights
        style_blend_weights = [1.0/len(style_images) for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight/total_blend_weight
                               for weight in style_blend_weights]

    # TODO: change checkpoint naming convention - they should also include hierarchy level
    if options.checkpoint_output and ("%s" not in options.checkpoint_output):
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain `%s` (e.g. `foo%s.jpg`)")

    print("\n>>> OUTPUT: %s\n" % (options.output))

    total_time = time.time()
    
    hierarchy_counter = 1

    ITER_DIVIDER_BASE = 1.5
    iter_divider = ITER_DIVIDER_BASE
    iter_hierarchy = [ options.iterations ]
    
    dim_first = (content_image.shape[0], content_image.shape[1])
    dim_hierarchy = [ dim_first ]
    dim_divider = 2
    dim_min = dim_first[0] if dim_first[0] < dim_first[1] else dim_first[1]
    while dim_min > 128 and hierarchy_counter < options.max_hierarchy:
        dim_new = tuple(x // dim_divider for x in dim_first)
        dim_hierarchy.append(dim_new)
        iter_hierarchy.append(int(options.iterations / iter_divider))
        
        dim_min = dim_new[0] if dim_new[0] < dim_new[1] else dim_new[1]
        
        dim_divider = dim_divider * 2
        iter_divider = iter_divider * ITER_DIVIDER_BASE
        hierarchy_counter = hierarchy_counter + 1

    num_channels = content_image.shape[2]
    h_initial_guess = content_image

    h_content = content_image
    
    # If noiseblend is not specified, it should be 0.0
    if options.initial_noiseblend is None:
        options.initial_noiseblend = 0.0
    
    hierarchy_steps = len(dim_hierarchy)
    for idx in reversed(range(hierarchy_steps)):
        dim = dim_hierarchy[idx]
        iter = iter_hierarchy[idx]
        
        # There is no point of getting below 25 iterations
        if hierarchy_steps > 1 and iter < 25:
            iter = 25
        
        is_last_hierarchy_level = (idx == 0)
        
        # x == dim[1], y == dim[0], meh
        print("Processing: %s / %d" % ((dim[1], dim[0]),iter))
        
        # If we only do 1 hierarchy step (e.g. no multgrid) - we don't need to resize content/initial
        if options.max_hierarchy > 1:
            h_initial_guess = scipy.misc.imresize(h_initial_guess, (dim[0], dim[1], num_channels))
            h_content = scipy.misc.imresize(content_image, (dim[0], dim[1], num_channels))
        
        coeff = 0.9
        h_initial_guess = h_initial_guess * coeff + h_content * (1.0 - coeff)
        
        target_shape = h_content.shape
        h_style_images = []
        for i in range(len(style_images)):
            style_scale = STYLE_SCALE
            if options.style_scales is not None:
                style_scale = options.style_scales[i]
            h_style_images.append( scipy.misc.imresize(style_images[i], style_scale *
                    target_shape[1] / style_images[i].shape[1]) )
        
        h_preserve_colors_coeff = 0.0
        if is_last_hierarchy_level:
            if options.preserve_colors == 'all' or options.preserve_colors == 'out':
                h_preserve_colors_coeff = 1.0
        else:
            if options.preserve_colors == 'all' or options.preserve_colors == 'interm':
                #h_preserve_colors_coeff = 0.5
                # we want biggest step to have least color preservation, to get proper style coloring, alpha = 1.0 on biggest step
                # -2 is due to idx starting from 0 and last layer not obeying this scheme
                h_preserve_alpha = (hierarchy_steps - 1 - idx) / (hierarchy_steps - 2)
                SMALLEST_STEP_PC = 1.0
                BIGGEST_STEP_PC = 0.3
                h_preserve_colors_coeff = BIGGEST_STEP_PC * h_preserve_alpha + SMALLEST_STEP_PC * (1.0 - h_preserve_alpha)
        
        #print("Preserve colors coeff: %f" % (h_preserve_colors_coeff))
        
        for iteration, image in stylize(
            network_file=options.network_file,
            network_type=options.network_type,
            initial=h_initial_guess,
            #initial=None,
            initial_noiseblend=options.initial_noiseblend,
            content=h_content,
#            styles=style_images,
            styles=h_style_images,
            preserve_colors_coeff=h_preserve_colors_coeff,
            preserve_colors_prior=(options.preserve_colors=='before'),
            iterations=iter,
            content_weight=options.content_weight,
            content_weight_blend=options.content_weight_blend,
            style_weight=options.style_weight,
            style_distr_weight=options.style_distr_weight,
            style_layer_weight_exp=options.style_layer_weight_exp,
            style_blend_weights=style_blend_weights,
            style_feat_type=options.style_feat_type,
            tv_weight=options.tv_weight,
            learning_rate=options.learning_rate,
            beta1=options.beta1,
            beta2=options.beta2,
            epsilon=options.epsilon,
            ashift=options.ashift,
            pooling=options.pooling,
            optimizer=options.optimizer,
            print_iterations=options.print_iterations,
            checkpoint_iterations=options.checkpoint_iterations
        ):
            output_file = None
            combined_rgb = image
            if iteration is not None:
                if options.checkpoint_output:
                    checkpoint_filename = options.checkpoint_output % ("%04dx%04d-%04d" % (dim[0], dim[1], iteration))
                    comimg.imsave(checkpoint_filename, combined_rgb)
            else:
                h_initial_guess = image
                    
        if is_last_hierarchy_level:
            if options.no_collage is None or options.no_collage == False:
                # For now, only works with the first style
                combined_rgb, _ =  build_collage.build_collage(
                                        np.clip(combined_rgb, 0, 255).astype(np.uint8),
                                        np.clip(content_image, 0, 255).astype(np.uint8),
                                        np.clip(style_images[0], 0, 255).astype(np.uint8),
                                        'crop'
                                        )
            
            # Last hierarchy level, we have the final output
            comimg.imsave(options.output, combined_rgb)
        else:
            # Not the last hierarchy level
            # True to save intermediate hierarchy shots
            if False:
                h_intermediate_name = "h_interm_%04dx%04d.jpg" % (dim[0], dim[1])
                comimg.imsave(h_intermediate_name, h_initial_guess)

        # True to save scaled content images
        if False:
            h_content_name = "h_content_%04dx%04d.jpg" % (dim[0], dim[1])
            comimg.imsave(h_content_name, h_content)
        
        # True to save scaled style images
        if False:
            for i in range(len(h_style_images)):
                h_style_name = "h_style%d_%04dx%04d.jpg" % (i, dim[0], dim[1])
                comimg.imsave(h_style_name, h_style_images[i])
      

    #if options.output:
    #    comimg.imsave(options.output, h_initial_guess)
      
    print("Total time: %fs" % (time.time() - total_time))

if __name__ == '__main__':
    main()
