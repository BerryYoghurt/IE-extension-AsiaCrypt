# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image

import time
import os, sys, platform
import copy as cp
import math
import numpy as np
from PIL import Image

import time
import os, sys, platform
import copy as cp
import math

from image_encryption.utils import *
from image_encryption.clip_helper import *
from image_encryption.crypto import *
np.set_printoptions(threshold=sys.maxsize)


N = 256
mask_bit = 14
num_mask = 64
mask_bar = 16
#  need to send at least 8 macs, because MCU read is 16 line per read


auth_expansion = 4
"""randomness"""
rd_expansion = 4
rd_height = 8
rd_width = 16
rd_width3 = 18


# === CONSTANTS ===
FEATURE_NUM_BYTES = 1024  # Number of bytes for features
MAC_NUM_BYTES = 32        # Number of bytes for MAC
BIT_MASK = 255            # Used for bit operations
SUBSAMPLING_420 = 2       # JPEG 4:2:0 subsampling

# Use existing parameter names for these if already defined, else define here
"""MAC"""
# AUTH_HEIGHT = 32
# Now they are 1024 bytes of features and 32 bytes for the MAC = 1056 bytes
# 8448 bits. I want to limit that to 400 squares per row.
AUTH_SIZE = (FEATURE_NUM_BYTES + MAC_NUM_BYTES) * 8
"""Repeating code"""
EXPANSION = 4
# We need to make sure the final image width and height are multiples of 16.
# If we guarantee that the image width is always larger than the lossless part (EXPANSION * WIDTH), then we only care about the height.
# The larger lossless part (auth) is 8448 bits. After duplication it will be EXPANSION * AUTH_SIZE.
# 128 pixels = 8 MCUs = 32 rows of information
AUTH_HEIGHT = 31   # 31 rows
RD_HEIGHT = 1       # only one row
EXPANDED_AUTH_HEGIHT = AUTH_HEIGHT * EXPANSION
EXPANDED_RD_HEGIHT = RD_HEIGHT * EXPANSION
"""Randomness"""
RD_SIZE = 16 * 8


from image_encryption.utils import  *
from image_encryption.clip_helper import *
from image_encryption.crypto import *

#from utils import  *
#from clip_helper import *
#from crypto import *
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
original_fn = "bear.jpg"


"""
option1: median filter
option2: salt and pepper denoise algorithm 1 in
https://www.frontiersin.org/articles/10.3389/fams.2022.918357/full#T2
"""

    
def pre_process(im):
    width, height = im.size # for image array in later reference, it is (height, width)
    # w16, h16 = math.ceil(width/16.0)*16, math.ceil(height/16.0)*16
    # newim = im.resize((w16,h16))
    # Resize to something small for google photos
    if width > 1000 or height > 1000:
        max_ = max(width, height)
        ratio = max_/1000.0
        width /= ratio
        height /= ratio
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
    
def separate_img(tag_ctxt):
    rd_ct_height = EXPANDED_RD_HEGIHT
    # HEIGHT is predetermined, and width is just what is left from encoding 
    # RD_SIZE * EXPANSION * EXPANSION as 3 color components for height RD_HEGIT
    rd_ct_width = int(math.ceil(RD_SIZE / (3 * RD_HEIGHT))) * EXPANSION 
    auth_ct_height = EXPANDED_AUTH_HEGIHT
    auth_ct_width  = int(math.ceil(AUTH_SIZE / (3 * AUTH_HEIGHT))) * EXPANSION 
    randomness = tag_ctxt[:rd_ct_height,:rd_ct_width]
    randomness = deduplicate_pixel_array(randomness, EXPANSION)
    r = pixel_array_to_bit_array(randomness)
    # randomness = subsamp(randomness,rd_expansion//2)
    # randomness = randomness[:, EXPANSION//2::EXPANSION][EXPANSION//2::EXPANSION, :]
    r = (randomness > 128).astype(np.uint8)
    tag_ctxt = tag_ctxt[rd_ct_height:]
    raw_tags = tag_ctxt[:auth_ct_height,:auth_ct_width]
    raw_tags = deduplicate_pixel_array(raw_tags, EXPANSION)
    # raw_tags = subsamp(raw_tags22,auth_expansion)
    tags = pixel_array_to_bit_array(raw_tags)
    # tags = (raw_tags > 128).astype(np.uint8)
    ctxt = tag_ctxt[auth_ct_height:]
    return ctxt,tags, r


def send(pt, key, mac_key, feature, q, output_filename):
    """
    Packs, encrypts, and authenticates the image and features for secure transmission.
    Each step is paired with a reverse step in recv().
    """
    # 1. Generate randomness for one-time keys (reverse of randomness extraction in recv)
    randomness_im = os.urandom(16)
    print("Randomness for image: ", randomness_im)
    # Each byte in randomness_im is XORed with 255 to generate another randomness value for the MAC key
    randomness_mac = b''.join([int.to_bytes(255 ^ x, length=1, byteorder="big", signed=False) for x in randomness_im])
    im_one_time_key = gen_short_one_time_key(randomness_im, key)
    mac_one_time_key = gen_short_one_time_key(randomness_mac, key)

    # 2. Encrypt the image (reverse of image decryption in recv)
    im_mod_add_key = gen_long_one_time_key(pt.shape, im_one_time_key)
    ctxt = sub_samp_image(encrypt_mod(pt, im_mod_add_key), "420")
    # print("Size of ciphertext: ", ctxt.shape)
    # Encode the features as bytes
    # I will treat it as  16 strings, each of size 528 bits
    # Therefore:

    # 3. Encode and MAC the features
    feature_bytes = feature.tobytes()
    mac = gen_hmac(mac_key, feature_bytes)
    # Append the mac to the features
    auth = feature_bytes + mac
    # We need to convert auth to a numpy array of uints
    auth = np.array([int.from_bytes(auth[i:i+1], byteorder="big", signed=False) for i in range(len(auth))], dtype='uint8')
    # auth = auth.reshape((AUTH_HEIGHT, (FEATURE_NUM_BYTES + MAC_NUM_BYTES) // AUTH_HEIGHT))
    print(f"Shape of auth before encryption {auth.shape}")
    mac_mod_add_key = gen_long_one_time_key(auth.shape, mac_one_time_key)
    print(f"Shape of mac_mod_add_key = {mac_mod_add_key.shape}")
    auth = auth ^ mac_mod_add_key

    # 4. Prepare MAC+feature array for embedding
    auth = auth.tobytes()
    print(f"len of auth bytearray {len(auth)}")
    auth_bit_array = bytearray_to_bit_array(auth)
    auth_bit_matrix = reshape_bit_array_to_3d(auth_bit_array, AUTH_HEIGHT)
    auth_pixel_matrix = bit_array_to_pixel_array(auth_bit_matrix)
    auth_pixel_matrix = duplicate_pixel_array(auth_pixel_matrix, EXPANSION)
    print(f"Dimension of auth {auth_pixel_matrix.shape}")
    
   

    # 5. Prepare randomness for embedding (reverse of randomness extraction in recv)
    randomness = randomness_im
    randomness_bit_array = bytearray_to_bit_array(randomness)
    randomness_bit_matrix = reshape_bit_array_to_3d(randomness_bit_array, RD_HEIGHT)
    randomness_pixel_matrix = bit_array_to_pixel_array(randomness_bit_matrix)
    randomness_pixel_matrix = duplicate_pixel_array(randomness_pixel_matrix, EXPANSION)
    print(f"Dimension of randomness {randomness_pixel_matrix.shape}")

    # 6. Append MAC+feature and randomness to the ciphertext image (reverse of separate_img in recv)
    ctxt_append = append_img(ctxt, auth_pixel_matrix, randomness_pixel_matrix)
    ctxtim = Image.fromarray(ctxt_append)

    # 7. Save the tagged ciphertext image (reverse of reading tagged image in recv)
    ctxtim.save(output_filename, quality=q, subsampling=SUBSAMPLING_420)
    

"""
Takes in
pt: array representing images
api: facebook api used for uploading and downloading
q:   quality factor
filt: flag to turn on/off filter. Default to True
"""
def expt_encrypt(pt,q, output_filename, key, mac_key):

    im = Image.fromarray(pt)
    t1 = time.perf_counter()
    orig_feature = clip_feature(im).cpu().detach().numpy().astype('float16')
    t2 = time.perf_counter()
    
    send(pt, key, mac_key,orig_feature,q, output_filename)
    return t2 - t1
   

def encrypt(quality_factor, input_filename, output_filename, key, mac_key):
    im = Image.open(input_filename)
    pt = pre_process(im)
    return expt_encrypt(pt,quality_factor, output_filename, key, mac_key)


def recv(password, filt):
    """
    Exact reverse of send(): Unpacks, decrypts, and verifies the image and features.
    Steps mirror send() in reverse order for clarity and correctness.
    """
    # 1. Read the tagged ciphertext image (reverse of ctxtim.save in send)
    if platform.system() == 'Windows' or not os.path.exists('./jpeg-9f'):
        local_ctxt_wtag = read_result()
        print("No jpeg-9f")
    else:
        os.system('./jpeg-9f/djpeg -bmp -nosmooth -outfile post420.bmp ctxt.jpeg')
        local_ctxt_wtag = djpeg_read_result()
        print("Using jpeg-9f")

    # 2. Separate the image into ciphertext, MACs, and randomness (reverse of append_img in send)
    local_ctxt, local_macs_bit_matrix, randomness_bit_matrix = separate_img(local_ctxt_wtag)

    # 3. Reconstruct randomness and keys
    r = flatten_bit_array(randomness_bit_matrix, RD_SIZE)
    r = bit_array_to_bytearray(r)
    randomness_im = r
    print("Extracted randomness for image: ", randomness_im)
    randomness_mac = b''.join([int.to_bytes(255 ^ x, length=1, byteorder="big", signed=False) for x in r])

    key, mac_key = gen_keys(password)
    im_one_time_key = gen_short_one_time_key(randomness_im, key)
    mac_one_time_key = gen_short_one_time_key(randomness_mac, key)

    # 4. Extract and decrypt MAC+feature array
    received_features_macs = flatten_bit_array(local_macs_bit_matrix, AUTH_SIZE)
    received_features_macs = bit_array_to_bytearray(received_features_macs)
    received_features_macs = np.array(
        [int.from_bytes(received_features_macs[i:i+1], byteorder="big", signed=False) for i in range(len(received_features_macs))],
        dtype='uint8'
    )
    mac_mod_add_key = gen_long_one_time_key(received_features_macs.shape, mac_one_time_key)
    received_features_macs = received_features_macs ^ mac_mod_add_key
    received_features_macs = received_features_macs.tobytes()
    received_features = received_features_macs[:FEATURE_NUM_BYTES]  # feature bytes
    received_macs = received_features_macs[FEATURE_NUM_BYTES:]      # mac bytes

    # 5. Verify MAC (reverse of gen_hmac in send)
    try:
        ver_hmac(mac_key, received_features, received_macs)
        ret = True
    except Exception:
        ret = False

    # 6. Decrypt image (reverse of encrypt_mod and sub_samp_image in send)
    im_mod_add_key = gen_long_one_time_key(local_ctxt.shape, im_one_time_key)
    local_pt = decrypt_mod(local_ctxt, im_mod_add_key)
    cleaned_pt = sub_samp_image(local_pt, "420", filt).astype('uint8')

    # 7. Extract features and compare (reverse of feature extraction in send)
    local_feature = get_feature(cleaned_pt, local_fn)
    received_features = np.frombuffer(received_features, dtype='float16')
    received_features = cp.copy(received_features)
    received_features = torch.from_numpy(received_features)
    sim = cos_sim(local_feature, received_features)
    print(ret, sim)
    # ret = ret and sim > 0.9
    return ret, cleaned_pt



def decrypt(password, filt):
    ret, cleaned_pt = recv(password,filt)
    return ret, Image.fromarray(cleaned_pt)


def gen_keys(password):
    ckdf = ConcatKDFHash(
        algorithm=hashes.SHA256(),
        length = 48,
        otherinfo=None
    )
    derived_key = ckdf.derive(str.encode(password))

    mac_key = derived_key[:32]
    key = derived_key[32:]
    return key, mac_key

#key, mac_key = gen_keys('hhh')
#encrypt(70, original_fn, 'ctxt.jpeg', key, mac_key)
#ret, image = decrypt('hhh', True)
#image.save("decrypted.jpeg", quality=90)
