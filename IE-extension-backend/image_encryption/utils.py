import numpy as np
import scipy.signal
from PIL import Image
import math
import copy as cp
from jpeglib import read_dct, from_dct, read_spatial

def expand(p,num):
    # expand array p into num x num larger.
    return np.repeat(np.repeat(p, repeats=num, axis=1), repeats=num, axis=0)

    
def subsamp(p,num):
    return p[:,0::num][0::num,:]
    
def append_img(ctxt,tags,randomness):
    coldiff = ctxt.shape[1] - randomness.shape[1]
    padding = np.zeros((randomness.shape[0],coldiff,randomness.shape[2]))
    r = np.hstack((randomness, padding))

    coldiff = ctxt.shape[1] - tags.shape[1]
    padding = np.zeros((tags.shape[0],coldiff,tags.shape[2]))    
    tg = np.hstack((tags,padding))

    tagged_ctxt = np.vstack((r,tg,ctxt))
    return tagged_ctxt.astype('uint8')
    



def sign(x):
    return np.where(x > 0, 1, 0)
    
def sign_dot(h, im):
    hash = sign(np.dot(h,im)).astype(int)
    return hash
    
def sp_alg(pt, size=5):
    res = scipy.signal.medfilt(pt,size)
    return res

def sp_alg_preserve(pt, size=5):
    res = scipy.signal.medfilt(pt,size)
    indices = np.where((res-pt>100) |  (res-pt<-100))
    pt[indices] = res[indices]
    return pt
    
def sub_sampling(pt, ratio="", med_filt=False):
    num_row, num_col = pt.shape
    if ratio == "411":
        sampled_pt = pt[:,0::4] # sample every 4th col
        if med_filt: row_sampled_pt = sp_alg_preserve(sampled_pt,5)
        restruct_pt = np.repeat(sampled_pt,repeats=4,axis=1)[:,:num_col:]
    elif ratio == "422":
        sampled_pt = pt[:,0::2] # sample every 2nd col
        if med_filt: row_sampled_pt = sp_alg_preserve(sampled_pt,5)
        restruct_pt = np.repeat(sampled_pt,repeats=2,axis=1)[:,:num_col:]
        
    elif ratio == "420":
        col_sampled_pt = pt[:,0::2] # sample every 2nd col
        row_sampled_pt = col_sampled_pt[0::2,:] # sample every 2nd row
        if med_filt: row_sampled_pt = sp_alg_preserve(row_sampled_pt,3)
        col_pt = np.repeat(row_sampled_pt,repeats=2,axis=1)[:,:num_col:]
        restruct_pt = np.repeat(col_pt,repeats=2,axis=0)[:num_row:,:]
        return restruct_pt
    else:
        # 4x2
        col_sampled_pt = pt[:,0::4] # sample every 4th col
        row_sampled_pt = col_sampled_pt[0::2,:] # sample every 2nd row
        if med_filt: row_sampled_pt = sp_alg_preserve(row_sampled_pt,5)
        col_pt = np.repeat(row_sampled_pt,repeats=4,axis=1)[:,:num_col]
        restruct_pt = np.repeat(col_pt,repeats=2,axis=0)[:num_row,:]
        
    return restruct_pt



def sub_samp_image(pt,ratio="",med_filt=False):
    # assume dimension of 3
    pt[:,:,0] =sub_sampling(pt[:,:,0], ratio, med_filt)
    pt[:,:,1] =sub_sampling(pt[:,:,1], ratio, med_filt)
    pt[:,:,2] =sub_sampling(pt[:,:,2], ratio, med_filt)
    return pt


"""Functions to encode parts of the message that we want to send losslessly"""
def bytearray_to_bit_array(byte_array):
  assert type(byte_array) == bytes
  """Converts a bytearray to a numpy array of bits, keeping leading zeros."""
  # Convert bytearray to a list of integers
  int_list = list(byte_array)
  # Convert each integer to its binary representation (as a string)
  binary_strings = [bin(x)[2:].zfill(8) for x in int_list]
  # Join the binary strings and convert to a list of integers (0 or 1)
  # bit_list = [int(bit) for binary_string in binary_strings for bit in binary_string]
  bit_list = [int(bit) for binary_string in binary_strings for bit in binary_string]
  # Convert the list of integers to a numpy array
  return np.array(bit_list)


def bit_array_to_bytearray(bit_array):
    """
    Converts a 1D numpy array of bits (0 or 1) back into a bytearray.

    Args:
        bit_array: A 1D numpy array of bits (0 or 1).

    Returns:
        A bytearray containing the original byte data.

    Raises:
        ValueError: If the number of bits is not a multiple of 8.
    """
    num_bits = bit_array.size

    if num_bits % 8 != 0:
        raise ValueError("The number of bits must be a multiple of 8 to convert to bytes without padding.")

    # Reshape the bit array into groups of 8 bits (bytes)
    byte_bits = bit_array.reshape(-1, 8)

    # Convert each 8-bit group into an integer (byte value)
    byte_values = []
    for byte in byte_bits:
        # Convert the array of 8 bits (e.g., [1, 0, 1, 0, 1, 0, 1, 0]) to a binary string ('10101010')
        binary_string = ''.join(map(str, byte))
        # Convert the binary string to an integer
        byte_value = int(binary_string, 2)
        byte_values.append(byte_value)

    # Convert the list of byte values into a bytearray
    return bytearray(byte_values)


def reshape_bit_array_to_3d(bit_array, height):
  """
  Reshapes a 1D numpy array of bits into a 3D array of shape (height, , 3).
  Filled color-first.

  Args:
    bit_array: A 1D numpy array of bits (0 or 1).

  Returns:
    A 3D numpy array of shape (total_bits // (height * 3), height, 3). If the total number of bits
    is not divisible by 3 or width, it is padded with trailing zeros.
  """
  total_bits = bit_array.size
  print(f"Total bits = {total_bits}")
  divisor = 3 * height
  rem = total_bits % divisor
  if rem != 0:
    padding = np.zeros(divisor - rem, dtype=np.int8)
    bit_array = np.concatenate([bit_array, padding])
    total_bits += divisor - rem

  # The total number of pixels is the total number of bits divided by 3 (for RGB)
  num_pixels = total_bits // 3

  width = num_pixels // height

  # Reshape the 1D array into a 3D array (H, W, 3)
  # The order='C' (default) or 'F' (Fortran) will determine how the array is read.
  # 'C' order fills row by row, then color channels within each row.
  # 'F' order fills column by column, then color channels within each column.
  # To fill color-first as requested, we need to arrange the 1D array
  # such that the first 3 elements are the R, G, B for the first pixel,
  # the next 3 for the second pixel, and so on.
  # Reshaping with shape (-1, 3) will group the bits into sets of 3 (pixels).
  # Then reshaping this into (H, W, 3) will arrange these pixels into the HxW grid.
  print(f"(H,W) = {(height, width)}")
  reshaped_3d = bit_array.reshape(-1, 3).reshape((height, width, 3))

  return reshaped_3d

def flatten_bit_array(bit_array, original_size):
    return bit_array.reshape(-1, 3).flatten()[:original_size]


def duplicate_pixel_array(pixel_array, num_duplicates):
    return np.repeat(np.repeat(pixel_array, num_duplicates, axis=0), num_duplicates, axis=1)

def deduplicate_pixel_array(pixel_array, num_duplicates):
    return pixel_array[num_duplicates//2::num_duplicates, num_duplicates//2::num_duplicates]

def bit_array_to_pixel_array(bit_array):
    return bit_array * 255

def pixel_array_to_bit_array(pixel_array):
    return (pixel_array > 128).astype(np.uint8)





"""read result from 'result.txt' that is written by djpeg"""
def djpeg_read_result():
    f= open('./result.txt','r')
    lines = f.readlines()
    v,w = math.ceil(int(lines[0])/16.0)*16, math.ceil(int(lines[1])/16.0)*16
    pt = np.zeros((v,w,3))
    group_size = 16
    group = -1
    comp = -1
    row = -1
    inx = -1
    for line in lines[2:]:
        if "Group" in line:
            group +=1
            comp = -1
            continue
        if "Component" in line:
            comp +=1
            row = -1
            continue
        if "Row" in line:
            row +=1
            inx = -1
            continue
        inx +=1
        #print(row,inx,comp)
        pt[group_size*group + row][inx][comp] = int(line)


    sml_row=  range(0, int(w/2))
    hw =int(w/2)
    if v <= 16:
        sml_col = range(0,8)
    else:
        # assume v is a multiple of 16
        lst = []
        for i in range(0,int(v/16)):
            lst+=list((range(i*16,i*16+8)))
        sml_col = cp.deepcopy(lst)

    # small image
    smallpt = np.zeros((int(v/2),int(w/2),3))
    smallpt[:,:,0] = pt[0::2,0::2,0]
    smallpt[:,:,1] = pt[:,:hw,1][sml_col,:]
    smallpt[:,:,2] = pt[:,:hw,2][sml_col,:]
    smallim = Image.fromarray(smallpt.astype('uint8'),mode='YCbCr').convert('RGB')

    # enlarge cb cr by 2x2
    cb = np.repeat(np.repeat(pt[:,:hw,1][sml_col,:],repeats=2,axis=1),repeats=2,axis=0)
    cr = np.repeat(np.repeat(pt[:,:hw,2][sml_col,:],repeats=2,axis=1),repeats=2,axis=0)
    pt[:,:,1] = cb
    pt[:,:,2]= cr
    im = Image.fromarray(pt.astype('uint8'),mode='YCbCr').convert('RGB')
    rgb_pt = np.array(im)
    return rgb_pt



def read_result(filename='ctxt.jpeg'):
    # print('inside read_result')
    im = read_dct(filename)
    # Avoid upsampling
    y = from_dct(Y=im.Y, qt=im.qt[0])
    cb = from_dct(Y=im.Cb, qt=im.qt[1])
    cr = from_dct(Y=im.Cr, qt=im.qt[-1])
    y.write_dct('Y_'+filename)
    cb.write_dct('Cb_'+filename)
    cr.write_dct('Cr_'+filename)
    # print('analyzed image')
    y = read_spatial(f'Y_{filename}', dither_mode=0).spatial
    cb = read_spatial(f'Cb_{filename}', dither_mode=0).spatial
    cr = read_spatial(f'Cr_{filename}', dither_mode=0).spatial
    print(f"Y channel shape: {y.shape}")
    print(f"Cb channel shape: {cb.shape}")
    print(f"Cr channel shape: {cr.shape}")
    if y.shape[0] > cb.shape[0]:
        # If subsampling happened
        # Take only 1/4 of the luminance
        y = y[::2,::2]
        print(f"Subsampled Y channel shape: {y.shape}")
        y = np.repeat(np.repeat(y, repeats=2, axis=1), repeats=2, axis=0)
        cb = np.repeat(np.repeat(cb, repeats=2, axis=1), repeats=2, axis=0)
        cr = np.repeat(np.repeat(cr, repeats=2, axis=1), repeats=2, axis=0)
    # Otherwise keep things as they are

    print(f"Y channel shape: {y.shape}")
    print(f"Cb channel shape: {cb.shape}")
    print(f"Cr channel shape: {cr.shape}")
    pt = np.stack((y, cb, cr), axis=-1)

    # enlarge cb cr by 2x2
    im = Image.fromarray(pt.astype('uint8'),mode='YCbCr').convert('RGB')
    rgb_pt = np.array(im).astype('uint8')
    return rgb_pt
