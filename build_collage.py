# Copyright (c) 2017 Andrey Voroshilov

import os

import numpy as np
import scipy.misc
import time

import math
from argparse import ArgumentParser

from PIL import Image

import common_images as comimg

# default arguments
MODE = 'scale'

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--in',             dest='input', help='input file, should be of format tile_<content>_<style>*.jpg', metavar='INPUT', required=True)
    parser.add_argument('--mode',           dest='mode', help='mode to generate style preview, scale/crop/auto (default %(default)s)', metavar='MODE', default=MODE)
    parser.add_argument('--styles-path',    dest='styles_path', help='optional path to styles images (by default uses input base path)')
    parser.add_argument('--content-path',   dest='content_path', help='optional path to content images (by default uses input base path)')
    return parser


def build_collage(result_image, content_image, style_image, mode='crop'):
    # X and Y are swapped in PIL: size is (y, x, channels) - majority issue
    result_size = (result_image.shape[1], result_image.shape[0])
    
    collage_shape = (result_size[0], int(math.ceil(result_size[1] * 1.5)))
    
    collage = Image.new('RGB', collage_shape)
    # Result/stylized
    #####################################################
    collage.paste(Image.fromarray(result_image), (0, 0))
    
    # Content
    #####################################################
    content_size = (int(result_size[0] * 0.5), int(result_size[1] * 0.5))
    # X and Y are swapped again
    content_image = comimg.imresize(content_image, (content_size[1], content_size[0]))
    collage.paste(Image.fromarray(content_image), (0, result_size[1]))

    # Style
    #####################################################
    style_target_size = content_size
    style_target_aspect = style_target_size[0] / style_target_size[1]
    # X and Y are swapped in PIL: size is (y, x, channels)
    style_size = (style_image.shape[1], style_image.shape[0])
    style_aspect = style_size[0] / style_size[1]
    
    if mode == 'auto':
        EPS = 1e-1
        if abs(style_aspect / style_target_aspect - 1.0) < EPS:
            print("Selecting scale mode")
            mode = 'scale'
        else:
            print("Selecting crop mode")
            mode = 'crop'
    
    suffix = '_comb'
    if mode == 'scale':
        # SCALE mode
        #####################################################
        
        # check which dimension is dominant
        scale = 1.0
        if style_aspect > style_target_aspect:
            # X
            scale = style_target_size[0] / style_size[0]
        else:
            # Y
            scale = style_target_size[1] / style_size[1]
        style_final_size = [int(dim * scale) for dim in style_size]
        
        # X and Y are swapped again
        style_image = comimg.imresize(style_image, (style_final_size[1], style_final_size[0]))
        collage.paste(Image.fromarray(style_image), (content_size[0], result_size[1]))
        suffix = '_combs'
    else:
        # CROP mode
        #####################################################
       
        # check which dimension is dominant
        # Selecting MINIMAL dimension
        if style_aspect > style_target_aspect:
            # X
            scale = style_target_size[1] / style_size[1]
        else:
            # Y
            scale = style_target_size[0] / style_size[0]
        style_final_size = [int(dim * scale) for dim in style_size]
        
        # X and Y are swapped again
        style_image = comimg.imresize(style_image, (style_final_size[1], style_final_size[0]))
        style_size_new = (style_image.shape[1], style_image.shape[0])
       
        # Crop parts of MAXIMAL dimension
        x_offset = (style_size_new[0] - content_size[0]) / 2.0
        y_offset = (style_size_new[1] - content_size[1]) / 2.0
        style_image_crop = Image.fromarray(style_image).crop((x_offset, y_offset, style_size_new[0] - x_offset, style_size_new[1] - y_offset))
        collage.paste(style_image_crop, (content_size[0], result_size[1]))
        suffix = '_combc'    
    
    return np.array(collage), suffix
    
def main():
    parser = build_parser()
    options = parser.parse_args()

    build_time = time.time()
    
    original_image = comimg.imread(options.input)
    
    base_path_len = options.input.rfind('\\')
    base_path = options.input[0:base_path_len+1]
    processed = options.input[base_path_len+1:]
    
    print("Path: %s, file: %s" % (base_path, processed))

    PREFIXES = ['tiles_', 't_']
    for i in range(len(PREFIXES)):
        str_idx = processed.find(PREFIXES[i])
        if str_idx == 0:
            processed = processed[len(PREFIXES[i]):]
            break
    
    content_name, processed = comimg.trim_starting_filename(processed)
    # remove the underscore after the content filename
    print(processed)
    processed = processed[1:]

    print(processed)
    
    style_name, processed = comimg.trim_starting_filename(processed)
    # remove the underscore after the style filename
    processed = processed[1:]
    
    print("Content: %s\nStyle: %s" % (content_name, style_name))
    
    if options.content_path is None:
        content_image = comimg.imread(base_path + content_name)
    else:
        content_image = comimg.imread(options.content_path + content_name)
    
    if options.styles_path is None:
        style_image = comimg.imread(base_path + style_name)
    else:
        style_image = comimg.imread(options.styles_path + style_name)

    collage, suffix = build_collage(original_image, content_image, style_image, options.mode)
    
    output_name = comimg.add_suffix_filename(options.input, suffix)
    print("Out: %s" % (output_name))
    comimg.imsave(output_name, np.array(collage))

    print("Collage build time: %fs" % (time.time() - build_time))


if __name__ == '__main__':
    main()