# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import os

import numpy as np
import scipy.misc
import time

import math
from argparse import ArgumentParser

from PIL import Image

# default arguments
MODE = 'yuv'

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--in',
            dest='input', help='input image',
            metavar='INPUT')
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT')
    parser.add_argument('--stylized',
            dest='stylized', help='one or more style images',
            metavar='STYLIZED')
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT')
    parser.add_argument('--mode',
            dest='mode', help='recombination mode: yuv/hsv (default %(default)s)',
            metavar='MODE', default=MODE)           
    return parser


def lumatransfer(original_image, styled_image):
    #The luminosity transfer steps:
    # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
    # 2. Convert stylized grayscale into YUV (YCbCr)
    # 3. Convert original image into YUV (YCbCr)
    # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
    # 5. Convert recombined image from YUV back to RGB
    
    perform_grayscale = False
    if perform_grayscale == True:
        # 1
        styled_grayscale = rgb2gray(styled_image)
        styled_grayscale_rgb = gray2rgb(styled_grayscale)
        # 2
        styled_grayscale_yuv = np.array( Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr') )
    else:
        # Skip #1 and just convert to YUV
        styled_grayscale_yuv = np.array( Image.fromarray(styled_image.astype(np.uint8)).convert('YCbCr') )

    # 3
    original_yuv = np.array( Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr') )
    
    # 4
    w, h, _ = original_image.shape
    combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
    combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
    combined_yuv[..., 1] = original_yuv[..., 1]
    combined_yuv[..., 2] = original_yuv[..., 2]
    
    # 5
    img_out = np.array( Image.fromarray(combined_yuv, 'YCbCr').convert('RGB') )

    return img_out

def lumatransfer_hsv(original_image, styled_image):
    #The luminosity transfer steps:
    # 1. Convert stylized image into HSV
    # 2. Convert original image into HSV
    # 3. Recombine (originalHSV.H, element-min(originalHSV.s, stylizedHSV.S), stylizedHSV.V), BUT
    #   element-wise minimum for S is needed to avoid oversaturation
    # 4. Convert recombined image from HSV back to RGB
    
    # 1
    styled_hsv = np.array( Image.fromarray(styled_image.astype(np.uint8)).convert('HSV') )

    # 2
    original_hsv = np.array( Image.fromarray(original_image.astype(np.uint8)).convert('HSV') )
    
    # 3
    w, h, _ = original_image.shape
    combined_hsv = np.empty((w, h, 3), dtype=np.uint8)
    combined_hsv[..., 0] = original_hsv[..., 0] # H
    
    combined_hsv[..., 1] = np.minimum(original_hsv[..., 1], styled_hsv[..., 1])
    
    combined_hsv[..., 2] = styled_hsv[..., 2] # V
    
    # 4
    img_out = np.array( Image.fromarray(combined_hsv, 'HSV').convert('RGB') )

    return img_out    

def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

def rgb2gray(rgb):
    # Rec.601 luma
    #return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    # Rec.709 luma
    return np.dot(rgb[...,:3], [0.2126, 0.7152, 0.0722])

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb

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

    if options.input is not None:
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
        
        content_name, processed = trim_starting_filename(processed)
        # remove the underscore after the content filename
        print(processed)
        processed = processed[1:]

        print(processed)
        
        style_name, processed = trim_starting_filename(processed)
        # remove the underscore after the style filename
        processed = processed[1:]
        
        print("Content: %s\nStyle: %s" % (content_name, style_name))
        
        original_image_path = content_name
        styled_image_path = options.input
    else:
        original_image_path = options.content
        styled_image_path = options.stylized
    
    original_image = imread(original_image_path)
    styled_image = imread(styled_image_path)

    luma_time = time.time()

    suffix = '_pc'
    if options.mode == 'hsv':
        suffix = '_pchsv'
        img_out = lumatransfer_hsv(original_image=original_image, styled_image=styled_image)
    else:
        suffix = '_pcyuv'
        img_out = lumatransfer(original_image=original_image, styled_image=styled_image)
    
    print("Luma transfer time: %fs" % (time.time() - luma_time))

    if options.output is not None:
        output_filename = options.output
    else:
        output_filename = add_suffix_filename(options.input, suffix)
    print("Out: %s" % (output_filename))
    imsave(output_filename, img_out)
    
if __name__ == '__main__':
    main()