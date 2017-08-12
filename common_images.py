import numpy as np
import scipy.misc

from PIL import Image

def imread(path):
    img = scipy.misc.imread(path)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]        
    return img

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

EXTENSIONS = [ '.jpg', '.bmp', '.png', '.tga' ]
    
def trim_starting_filename(str):
    min_ext_pos = len(str)
    min_ext_idx = -1
    for i in range(len(EXTENSIONS)):
        filename_ext = str.find(EXTENSIONS[i])
        if filename_ext != -1 and filename_ext < min_ext_pos:
            min_ext_pos = filename_ext
            min_ext_idx = i

    filename = str
    if min_ext_idx != -1:
        filename = str[0:min_ext_pos + len(EXTENSIONS[min_ext_idx])]

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
