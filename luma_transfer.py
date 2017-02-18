# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import os

import numpy as np
import scipy.misc
import time

import math
from argparse import ArgumentParser

from PIL import Image

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', required=True)
    parser.add_argument('--stylized',
            dest='stylized', help='one or more style images',
            metavar='STYLIZED', required=True)
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=True)
    return parser


def lumatransfer(original_image, styled_image):
    #The luminosity transfer steps:
    # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
    # 2. Convert stylized grayscale into YUV (YCbCr)
    # 3. Convert original image into YUV (YCbCr)
    # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
    # 5. Convert recombined image from YUV back to RGB
    
    # 1
    styled_grayscale = rgb2gray(styled_image)
    styled_grayscale_rgb = gray2rgb(styled_grayscale)
    # 2
    styled_grayscale_yuv = np.array( Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr') )

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
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb

def main():
    parser = build_parser()
    options = parser.parse_args()

    original_image = imread(options.content)
    styled_image = imread(options.stylized)

    luma_time = time.time()

    img_out = lumatransfer(original_image=original_image, styled_image=styled_image)
    
    print("Luma transfer time: %fs" % (time.time() - luma_time))

    output_file = options.output
    imsave(options.output, img_out)
    
if __name__ == '__main__':
    main()