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
MODE = 'yuv'

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--in',         dest='input', help='input image', metavar='INPUT')
    parser.add_argument('--content',    dest='content', help='content image', metavar='CONTENT')
    parser.add_argument('--stylized',   dest='stylized', help='one or more style images', metavar='STYLIZED')
    parser.add_argument('--output',     dest='output', help='output path',metavar='OUTPUT')
    parser.add_argument('--mode',       dest='mode', help='recombination mode: yuv/hsv (default %(default)s)', metavar='MODE', default=MODE)           
    parser.add_argument('--content-path', dest='content_path', help='optional path to content images (by default uses input base path)')
    return parser

def histmatch_binned_ch(source_channel, reference_channel, num_bins=64):
    #Histogram matching steps:
    # 1. Calculate histogram for source and reference channels
    # 2. Calculate Cumulative Distribution Functions (CDFs) based on the histograms
    # 3. Map Source channel values to the CDF space using bins and calculated Source CDF
    #       if number of bins is lower than 256, use interpolation
    # 4. Map values from CDF space to the channel values, using Reference CDF
  
    src_ch_flatten = source_channel.ravel()
    ref_ch_flatten = reference_channel.ravel()
    
    # 1. Calculate histograms
    src_hist, src_hist_bins = np.histogram(src_ch_flatten, bins=num_bins, range=None, density=False)
    ref_hist, ref_hist_bins = np.histogram(ref_ch_flatten, bins=num_bins, range=None, density=False)

    # 2. Calculate normalized Cumulative Distribution Functions
    src_cdf = np.cumsum(src_hist).astype(np.float32)
    src_cdf *= 255.0 / src_cdf[-1]

    ref_cdf = np.cumsum(ref_hist).astype(np.float32)
    ref_cdf *= 255.0 / ref_cdf[-1]

    # 3. Source image channel values to the histogram CDF
    src_to_cdf = np.interp(src_ch_flatten, src_hist_bins[:-1], src_cdf)
    # 4. Histogram CDF to reference channel values
    cdf_to_ref = np.interp(src_to_cdf, ref_cdf, ref_hist_bins[:-1])
    
    return np.clip(cdf_to_ref, 0, 255).reshape(source_channel.shape)
    
def histmatch_ch(source_channel, reference_channel):
    #Histogram matching steps:
    # 1. Calculate histogram for source and reference channels
    # 2. Calculate Cumulative Distribution Functions (CDFs) based on the histograms
    # 3. Map Source channel values to the CDF space using bins and calculated Source CDF
    # 4. Map values from CDF space to the channel values, using Reference CDF
   
    src_ch_flatten = source_channel.ravel()
    ref_ch_flatten = reference_channel.ravel()
    
    # 1. Calculate histograms
    src_hist, _ = np.histogram(src_ch_flatten, bins=256, range=None, density=False)
    ref_hist, _ = np.histogram(ref_ch_flatten, bins=256, range=None, density=False)

    # 2. Calculate normalized Cumulative Distribution Functions
    src_cdf = np.cumsum(src_hist).astype(np.float32)
    src_cdf *= 255.0 / src_cdf[-1]

    ref_cdf = np.cumsum(ref_hist).astype(np.float32)
    ref_cdf *= 255.0 / ref_cdf[-1]

    # 3. Source image channel values to the histogram CDF
    src_to_cdf = src_ch_flatten
    src_to_cdf[...] = src_cdf[src_ch_flatten[...]]
    # 4. Histogram CDF to reference channel values
    cdf_to_ref = src_to_cdf
    cdf_to_ref[...] = np.clip(np.searchsorted(ref_cdf, src_to_cdf[...]) - 1, 0, 255)
    
    return cdf_to_ref.reshape(source_channel.shape)
    
def histmatch(original_image, styled_image, hsv=True):
    w, h, _ = styled_image.shape
    img_out = np.empty((w, h, 3), dtype=np.uint8)

    if hsv:
        styled_hsv = np.array( Image.fromarray(styled_image.astype(np.uint8)).convert('HSV') )
        original_hsv = np.array( Image.fromarray(original_image.astype(np.uint8)).convert('HSV') )

        for ch in range(3):
            img_out[..., ch] = histmatch_ch(styled_hsv[..., ch], original_hsv[..., ch])
       
        img_out = np.array( Image.fromarray(img_out, 'HSV').convert('RGB') )
    else:
        for ch in range(3):
            img_out[..., ch] = histmatch_ch(styled_image[..., ch], original_image[..., ch])
   
    return img_out


def lumatransfer(original_image, styled_image):
    #The luminosity transfer steps:
    # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
    # 2. Convert stylized grayscale into YUV (YCbCr)
    # 3. Convert original image into YUV (YCbCr)
    # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
    # 5. Convert recombined image from YUV back to RGB
    
    if original_image.shape != styled_image.shape:
        original_image = scipy.misc.imresize(original_image, styled_image.shape)
    
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

    if original_image.shape != styled_image.shape:
        original_image = scipy.misc.imresize(original_image, styled_image.shape)
    
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

def colortransfer(original_image, styled_image, mode='yuv'):
    if mode == 'hsv':
        return lumatransfer_hsv(original_image=original_image, styled_image=styled_image)
    elif mode == 'hist':
        return histmatch(original_image=original_image, styled_image=styled_image)
    else:
        return lumatransfer(original_image=original_image, styled_image=styled_image)
    

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

def main():
    parser = build_parser()
    options = parser.parse_args()

    if options.input is not None:
        base_path, processed = os.path.split(options.input)

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
            content_image_path = os.path.join(base_path, content_name)
        else:
            content_image_path = options.content_path + content_name
        
        styled_image_path = options.input
    else:
        content_image_path = options.content
        styled_image_path = options.stylized
    
    content_image = comimg.imread(content_image_path)
    styled_image = comimg.imread(styled_image_path)

    content_image_size = (content_image.shape[1], content_image.shape[0])
    styled_image_size = (styled_image.shape[1], styled_image.shape[0])
    
    collage_image = None
    if content_image_size[0] == styled_image_size[0] and math.fabs(content_image_size[1]*1.5 - styled_image_size[1]) < 2.0:
        collage_image = styled_image
        styled_image = np.array(Image.fromarray(styled_image).crop((0, 0, content_image_size[0], content_image_size[1])))
    
    luma_time = time.time()
    
    suffix = '_pc'
    if options.mode == 'hsv':
        suffix = '_pchsv'
    elif options.mode == 'hist':
        suffix = '_pchist'
    else:
        suffix = '_pcyuv'
        
    img_out = colortransfer(original_image=content_image, styled_image=styled_image, mode=options.mode)
    
    print("Luma transfer time: %fs" % (time.time() - luma_time))

    if collage_image is not None:
        new_collage = Image.fromarray(collage_image)
        new_collage.paste(Image.fromarray(img_out), (0, 0))
        img_out = np.array(new_collage)
    
    if options.output is not None:
        output_filename = options.output
    else:
        output_filename = comimg.add_suffix_filename(styled_image_path, suffix)
    print("Out: %s" % (output_filename))
    comimg.imsave(output_filename, img_out)
    
if __name__ == '__main__':
    main()