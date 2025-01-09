import numpy as np
import scipy.signal
from PIL import Image
import math
import copy as cp
from jpeglib import read_dct, from_dct, read_spatial


def append_img(ctxt,tags,randomness):
    coldiff = ctxt.shape[1] - randomness.shape[1]
    padding = np.zeros((randomness.shape[0],coldiff,randomness.shape[2]))
    r = np.hstack((randomness, padding))

    coldiff = ctxt.shape[1] - tags.shape[1]
    padding = np.zeros((tags.shape[0],coldiff,tags.shape[2]))    
    tg = np.hstack((tags,padding))

    tagged_ctxt = np.vstack((r,tg,ctxt))
    return tagged_ctxt.astype('uint8')
    

def separate_img(tag_ctxt, length,size):
    randomness = tag_ctxt[:16,:12]
    randomness = randomness[::2,::2,:]
    r = 1*(randomness > 128)
    tag_ctxt = tag_ctxt[16:]
    raw_tags22 = tag_ctxt[:length,:size]
    raw_tags = raw_tags22[0::2,0::2,:]
    tags = 1*(raw_tags > 128)
    ctxt = tag_ctxt[length:]
    return ctxt,tags, r

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


def bstr_to_bytes(s):
    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')

def bytes_to_bstr(b):
    return bin(int.from_bytes(b,byteorder='big'))[2:]
    
    # turn a 2d list of 0/1 to a list of bytes
def bstr_to_bytes2d(arr):
    results = []
    for row in arr:
        bstr =''.join(str(x) for x in row)
        bytes = bstr_to_bytes(bstr)
        results.append(bytes)
    return results
    
    # turn a list of bytes to 2d list of 0/1. If number of bits is less than size, it will add leading 0
def bytes_to_bstr2d(arr, size):
    results = []
    for byte in arr:
        b_lst = list(bytes_to_bstr(byte).zfill(size))
        blst_int = [int(x) for x in b_lst]
        #print(len(blst_int))
        results.append(blst_int)
    return results



"""read result from 'result.txt' that is written by djpeg"""
def old_read_result():
    f= open('result.txt','r')
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
    # Take only 1/4 of the luminance
    y = read_spatial(f'Y_{filename}', dither_mode=0).spatial[::2,::2]
    cb = read_spatial(f'Cb_{filename}', dither_mode=0).spatial
    cr = read_spatial(f'Cr_{filename}', dither_mode=0).spatial
    y = np.repeat(np.repeat(y, repeats=2, axis=1), repeats=2, axis=0)
      
    cb = np.repeat(np.repeat(cb, repeats=2, axis=1), repeats=2, axis=0)
    cr = np.repeat(np.repeat(cr, repeats=2, axis=1), repeats=2, axis=0)

    pt = np.stack((y, cb, cr), axis=-1)

    # enlarge cb cr by 2x2
    im = Image.fromarray(pt.astype('uint8'),mode='YCbCr').convert('RGB')
    rgb_pt = np.array(im).astype('uint8')
    return rgb_pt
