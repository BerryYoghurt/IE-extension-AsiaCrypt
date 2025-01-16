# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image

import time
import os, sys, shutil, platform
import copy as cp
from statistics import mean
import math
import random
import matplotlib.pyplot as plt
from itertools import chain
from image_encryption.utils import  *
from image_encryption.clip_helper import *
from image_encryption.crypto import *

np.set_printoptions(threshold=sys.maxsize)

"""Parameters"""

""" Make pixel value at least 30, at most 225 to avoid 'edge flip' """
pix_min = 70
pix_max = 205

# pix_min = 0
# pix_max = 255
glob_ctxt = ""

page_token = "Anonymized"    # long term page token
"""image"""

dct_range = 512

#small_im_size = (2, 2) # In convention of PIL: (width, height)
api = ""
ctxt_fn = "ctxt.jpeg"
local_fn = "local.jpeg"
fb_ctxt_fn = "fb_ctxt.jpeg"
fb_pt_fn = "fb.jpeg"
original_fn = "color.jpeg"
num_runs = 1
N = 256
mask_bit = 14
num_mask = 64
mask_bar = 16
#  need to send at least 8 macs, because MCU read is 16 line per read
# macs_sent = 8
macs_sent = 16
mac_size = 256
mac_size3 = math.ceil(mac_size/3)*3
#h.update(b"helloooooo")
#sig = h.finalize()
# 32 byte
"""
option1: median filter
option2: salt and pepper denoise algorithm 1 in
https://www.frontiersin.org/articles/10.3389/fams.2022.918357/full#T2
"""

    
def pre_process(im):
    width, height = im.size # for image array in later reference, it is (height, width)
    w16, h16 = math.ceil(width/16.0)*16, math.ceil(height/16.0)*16
    newim = im.resize((w16,h16))
    arr = np.array(newim)
    belowinx = np.where(arr<=pix_min)
    aboveinx = np.where(arr > pix_max)
    new_arr = cp.deepcopy(arr)
    new_arr[belowinx]=pix_min
    new_arr[aboveinx]=pix_max
    return new_arr


"""using algorithm 1 in https://papers.nips.cc/paper_files/paper/2012/file/072b030ba126b2f4b2374f342be9ed44-Paper.pdf"""

# vector having shape (N,d)
def proj_vec (N, d):
    np.random.seed(100)
    L=1
    v = np.random.normal(0, 1, size=(N*L,d))
#    w = np.zeros(v.shape)
#    for i in range(0,L):
#        for j in range(0,N):
#            w[i*N+j] = v[i*N+j]
#            for k in range(0,j-1):
#                w[i*N+j] =  w[i*N+j] -  np.dot(w[i*N+k],w[i*N+k])*v[i*N+j]
#            w[i*N+j] = w[i*N+j]/np.linalg.norm(w[i*N+j])
    w=v
    return w
    
    # 64 masked hashes
def gen_mask(mask_bit, N):
    np.random.seed(0)
    masks = np.zeros((num_mask,N), dtype=int)
    for i in range(num_mask):
        # Initialize an array of zeros with length 256
        bit_string = np.zeros(N, dtype=int)
        # Randomly choose 14 indices to set to 1
        ones_indices = np.random.choice(N, mask_bit, replace=False)
        # Set the chosen indices to 1
        bit_string[ones_indices] = 1
        masks[i] = bit_string
    return masks
    
"""compute hash for ctxt"""
def mask_hash(ctxt,mask,vec):
    ctxt_h = ctxt.astype('int').flatten()
    ctxt_hash = sign(np.dot(vec, ctxt_h)).astype(int)
    ctxt_msk_hash = mask & ctxt_hash
    return ctxt_msk_hash

"""compute hash for clip feature"""
def mask_hash_clip(feature,mask,vec):
    hash = sign(np.dot(vec,feature)).astype(int)
    msk_hash = mask & hash
    return msk_hash
    


def send(pt,mac_key,feature,q, output_filename):
    randomness_im = os.urandom(16)
    # print("Randomness in encrypt", randomness_im)
    randomness_mac = b''.join([int.to_bytes(255 ^ x, length=1, byteorder="big", signed=False) for x in randomness_im])
    im_one_time_key = gen_short_one_time_key(randomness_im)
    # print(im_one_time_key[:5])
    mac_one_time_key = gen_short_one_time_key(randomness_mac)

    im_mod_add_key = gen_long_one_time_key(pt.shape, im_one_time_key)
    ctxt = sub_samp_image(encrypt_mod(pt,im_mod_add_key),"420")

    # Encode the features as bytes
    # Now they are 1024 bytes of features and 32 bytes for the MAC = 1056 bytes
    # 8448 bits
    # I will treat it as  16 strings, each of size 528 bits
    # Threfore:

    # MAC the features
    feature = feature.tobytes()
    mac = gen_hmac(mac_key, feature)
    # print(feature)
    # Append the mac to the features
    
    auth = feature + mac
    auth = np.array([int.from_bytes(auth[i:i+1], byteorder="big", signed=False) for i in range(len(auth))], dtype='uint8').reshape((16, 66))
    mac_mod_add_key = gen_long_one_time_key(auth.shape, mac_one_time_key)
    # print("MAC KEY: ", mac_mod_add_key)
    # print("Randomness mac: ", randomness_mac)
    auth = auth ^ mac_mod_add_key

    # 1024 + 32 bytes = 1056 --> (16, 528, 3), 
    auth_size3 = 528

    macs_bit = np.array(bytes_to_bstr2d(auth,auth_size3))*255

    macs_bit = macs_bit.reshape(16, auth_size3//3,3)
    macs_bit_22 = np.repeat(np.repeat(macs_bit,repeats=2,axis=1),repeats=2,axis = 0).astype('uint8')
    
    # Now they are 16 bytes of randomness
    # 128 bits --> (8, 6, 3)
    # Previously, we had mac_size be the size of one MAC, 256 bits
    # we had 16 macs in total
    # And mac_size3 = 258 bits
    # I will treat the randomness as 8 random strings, each of size 16 bits
    # Threfore:
    random_strings = 8
    randomness_size = 16
    randomness_size3 = 18
    randomness = randomness_im 
    # print("Randomness: ", randomness)
    # Rewrite randomness bytearray as 8 random strings, each of size 16 bits
    randomness = [randomness[i:i+2] for i in range(0, len(randomness), 2)]
    randomness_bit = np.array(bytes_to_bstr2d(randomness, randomness_size3)) * 255
    randomness_bit = randomness_bit.reshape(random_strings, randomness_size3//3, 3)
    # print("Randomness bit: ", randomness_bit)
    randomness_bit_22 = np.repeat(np.repeat(randomness_bit, repeats=2, axis=1), repeats=2, axis=0).astype('uint8')

    ctxt_append = append_img(ctxt, macs_bit_22, randomness_bit_22)
    # print(macs_bit_22.shape)
    ctxtim = Image.fromarray(ctxt_append)
    
    ctxtim.save(output_filename,quality=q,subsampling=2)
    

"""
Takes in
pt: array representing images
api: facebook api used for uploading and downloading
q:   quality factor
filt: flag to turn on/off filter. Default to True
"""
def expt_encrypt(pt,q, output_filename, mac_key):

    im = Image.fromarray(pt)
    t1 = time.perf_counter()
    orig_feature = clip_feature(im).cpu().detach().numpy()
    t2 = time.perf_counter()
    
    send(pt,mac_key,orig_feature,q, output_filename)
    return t2 - t1
   

def encrypt(quality_factor, input_filename, output_filename, mac_key):
    im = Image.open(input_filename)
    pt = pre_process(im)
    return expt_encrypt(pt,quality_factor, output_filename, mac_key)


def recv(mask,filt):
    if platform.system() == 'Windows' or not os.path.exists('./jpeg-9f'):
        local_ctxt_wtag = read_result()
        print("No jpeg-9f")
    else:
        os.system('./jpeg-9f/djpeg -bmp -nosmooth -outfile post420.bmp ctxt.jpeg')
        local_ctxt_wtag = old_read_result()
        print("Using jpeg-9f")
    local_ctxt,local_macs_bstr, randomness_bstr = separate_img(local_ctxt_wtag,32,352)
    r = b''.join(bstr_to_bytes2d(randomness_bstr.reshape(8, 18)[:,2:]))
    # print("Randomness: ", r)
    randomness_im = r
    randomness_mac = b''.join([int.to_bytes(255 ^ x, length=1, byteorder="big", signed=False) for x in r])

    im_one_time_key = gen_short_one_time_key(randomness_im)
    # print(im_one_time_key[:5])
    mac_one_time_key = gen_short_one_time_key(randomness_mac)
    
    mac_key = b'\x19V\xcc\xe3\xc8\xd6\xa2h\x82\x97\xc7\x9e\x83\x04\x15s\xfd\x06"<7\xa0US\xf1\xe9\xbf\xe1\x9eM\xbe\x94'
    
    received_features_macs = bstr_to_bytes2d(local_macs_bstr.reshape(16,528)[:,:])
    received_features_macs = np.array([[int.from_bytes(m[i:i+1], byteorder="big", signed=False) for i in range(len(m))] for m in received_features_macs], dtype='uint8').reshape((16, 66))
    mac_mod_add_key = gen_long_one_time_key(received_features_macs.shape, mac_one_time_key)
    # mac_one_time_key same
    # print("MAC KEY: ", mac_mod_add_key)
    # print("Randomness mac: ", randomness_mac)
    received_features_macs = received_features_macs ^ mac_mod_add_key
    received_features_macs = b''.join(bytes(list(x)) for x in received_features_macs)
    received_features = received_features_macs[:1024]
    received_macs = received_features_macs[1024:]
    try:
        ver_hmac(mac_key, received_features, received_macs) 
        ret = True   
    except:
        ret = False
    
    im_mod_add_key = gen_long_one_time_key(local_ctxt.shape, im_one_time_key)
    local_pt = decrypt_mod(local_ctxt,im_mod_add_key)
    if filt:
        cleaned_pt = sub_samp_image(local_pt,"420",True).astype('uint8')
    else:
        cleaned_pt = sub_samp_image(local_pt,"420",False).astype('uint8')

    local_feature = get_feature(cleaned_pt,local_fn)
    received_features = np.frombuffer(received_features, dtype='float16')
    received_features = cp.copy(received_features)
    received_features = torch.from_numpy(received_features)
    # print(received_features)
    sim = cos_sim(local_feature, received_features)
    print(ret, sim)
    ret = ret and sim > 0.9
    return ret, cleaned_pt



def decrypt(filt):
    # Long term key
    mask = gen_mask(mask_bit,N)
    ret, cleaned_pt = recv(mask,filt)
    return ret, Image.fromarray(cleaned_pt)


def gen_keys(im_height, im_width, password):
    if im_height % 16 != 0 or im_width % 16 != 0:
        raise ValueError("Dimensions need to be multiples of 16")
    
    ckdf = ConcatKDFHash(
        algorithm=hashes.SHA256(),
        length=32 + im_height * im_width * 3,
        otherinfo=None
    )
    derived_key = ckdf.derive(str.encode(password))

    mac_key = derived_key[:32]
    key = np.reshape(np.frombuffer(derived_key[32:], dtype='uint8'), (im_width, im_height, 3))
    return key, mac_key