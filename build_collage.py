# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import os

import numpy as np
import scipy.misc
import time

import math
from argparse import ArgumentParser

from PIL import Image

# default arguments
MODE = 'scale'

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--in',
            dest='input', help='input file, should be of format tile_<content>_<style>*.jpg',
            metavar='INPUT', required=True)
    parser.add_argument('--mode',
            dest='mode', help='mode to generate style preview, scale/crop/auto (default %(default)s)',
            metavar='MODE', default=MODE)
    return parser


def imread_uint8(path):
    img = scipy.misc.imread(path).astype(np.uint8)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

EXTENSIONS = [ '.jpg', '.bmp', '.png', '.tga' ]
    
def trim_starting_filename(str):
    for i in range(len(EXTENSIONS)):
        filename_ext = str.find(EXTENSIONS[i])
        if filename_ext != -1:
            filename = str[0:filename_ext + len(EXTENSIONS[i])]
            break
    # move pointer to after the filename
    str = str[len(filename):]
    return filename, str

def add_suffix_filename(filename, suffix):
    for i in range(len(EXTENSIONS)):
        filename_ext = filename.rfind(EXTENSIONS[i])
        if filename_ext != -1:
            filename_out = filename[0:filename_ext] + suffix + filename[filename_ext:]
            break
    return filename_out
    
def main():
    parser = build_parser()
    options = parser.parse_args()

    build_time = time.time()
    
    original_image = imread_uint8(options.input)
    # X and Y are swapped in PIL: size is (y, x, channels) - majority issue
    original_size = (original_image.shape[1], original_image.shape[0])

    base_path_len = options.input.rfind('\\')
    base_path = options.input[0:base_path_len]
    processed = options.input[base_path_len:]
    
    print("Path: %s, file: %s" % (base_path, processed))

    processed = processed.replace('tiles_', '')
    processed = processed.replace('t_', '')
    
    content_name, processed = trim_starting_filename(processed)
    # remove the underscore after the content filename
    processed = processed[1:]

    style_name, processed = trim_starting_filename(processed)
    # remove the underscore after the style filename
    processed = processed[1:]
    
    print("Content: %s\nStyle: %s" % (content_name, style_name))
    
    collage_shape = (original_size[0], int(math.ceil(original_size[1] * 1.5)))
    
    collage = Image.new('RGB', collage_shape)
    # Original
    #####################################################
    collage.paste(Image.fromarray(original_image), (0, 0))
    
    # Content
    #####################################################
    content_size = (int(original_size[0] * 0.5), int(original_size[1] * 0.5))
    content_image = imread_uint8(base_path + content_name)
    # X and Y are swapped again
    content_image = scipy.misc.imresize(content_image, (content_size[1], content_size[0]))
    collage.paste(Image.fromarray(content_image), (0, original_size[1]))

    # Style
    #####################################################
    style_target_size = content_size
    style_target_aspect = style_target_size[0] / style_target_size[1]
    style_image = imread_uint8(base_path + style_name)
    # X and Y are swapped in PIL: size is (y, x, channels)
    style_size = (style_image.shape[1], style_image.shape[0])
    style_aspect = style_size[0] / style_size[1]
    
    suffix = '_comb'
    mode = options.mode
    
    if mode == 'auto':
        EPS = 1e-1
        if abs(style_aspect / style_target_aspect - 1.0) < EPS:
            print("Selecting scale mode")
            mode = 'scale'
        else:
            print("Selecting crop mode")
            mode = 'crop'
    
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
        style_image = scipy.misc.imresize(style_image, (style_final_size[1], style_final_size[0]))
        collage.paste(Image.fromarray(style_image), (content_size[0], original_size[1]))
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
        style_image = scipy.misc.imresize(style_image, (style_final_size[1], style_final_size[0]))
        style_size_new = (style_image.shape[1], style_image.shape[0])
       
        # Crop parts of MAXIMAL dimension
        x_offset = (style_size_new[0] - content_size[0]) / 2.0
        y_offset = (style_size_new[1] - content_size[1]) / 2.0
        style_image_crop = Image.fromarray(style_image).crop((x_offset, y_offset, style_size_new[0] - x_offset, style_size_new[1] - y_offset))
        collage.paste(style_image_crop, (content_size[0], original_size[1]))
        suffix = '_combc'
    
    output_name = add_suffix_filename(options.input, suffix)
    print("Out: %s" % (output_name))
    imsave(output_name, np.array(collage))
    
    print("Collage build time: %fs" % (time.time() - build_time))
    
if __name__ == '__main__':
    main()