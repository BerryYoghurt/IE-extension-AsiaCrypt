from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.concatkdf import ConcatKDFHash
import numpy as np

def gen_hmac(key, bytes):
    h = hmac.HMAC(key, hashes.SHA256())
    h.update(bytes)
    mac = h.finalize()
    return mac
   
def ver_hmac(key,bytes,mac):
    h = hmac.HMAC(key, hashes.SHA256())
    h.update(bytes)
    result = h.verify(mac)
    return result

def gen_hmac_list(key, bytes_list):
    result = []
    for i in bytes_list:
        sig = gen_hmac(key,i)
        result.append(sig)
    return result
    
    # return true if at least {bar} of mac verifies under a value in data_arr
def ver_hmac_list(key, data_arr,mac_arr,bar):
    # print('ver_hmac_list')
    count = 0
    for mac in mac_arr:
        for data in data_arr:
            try:
                ver_hmac(key,data,mac)
                count +=1
                break
            except BaseException as e:
                # print(e)
                continue
    print(count)
    return count >= bar

""" Encrypt an image using xor with a random value"""
def encrypt_xor(pt, key):
    """
    pt: input plaintext that represents image array (any shape)
    key: input key for encrypt the image array
    ctxt_fn: input file name for storing encrypted picture
    """
    # compute encrypted image
    ctxt = (pt ^key)
    return ctxt

""" Decrypt an image by xoring the ciphertext with key"""
def decrypt_xor(ctxt,key):
    pt = (ctxt ^ key)
    #ptim.show()
    return pt

""" Encrypt an image by adding a random value then mod 256"""
def encrypt_mod(pt, key):
    """
    pt: input plaintext that represents image array (any shape)
    key: input key for encrypt the image array
    ctxt_fn: input file name for storing encrypted picture
    """
    # compute encrypted image
    if pt.dtype == 'uint8' and key.dtype == 'uint8':
        ctxt = (pt + key)
    else:
        ctxt = (pt + key) % 256

    return ctxt

""" Decrypt an image by xoring the ciphertext with key"""
def decrypt_mod(ctxt,key):
    pt = (ctxt - key)
    return pt


def gen_long_one_time_key(size, key):    
    # algorithm = algorithms.AES128(key) 
    algorithm = algorithms.AES(key)
    cipher = Cipher(algorithm, mode=modes.CTR(b'\x00'*16))
    encryptor = cipher.encryptor()
    num_bytes = 1
    for n in size:
        num_bytes *= n
    ct = encryptor.update(b"\x00"*num_bytes)
    return np.frombuffer(ct, dtype='uint8').reshape(size)

# 2. generate like Figure 6 with CTR mode starting from 0
# 3. GCM encrypted feature vector
# 4. Measure running time with and without authentication
def gen_short_one_time_key(randomness, long_term_key=None):
    if long_term_key == None:
        long_term_key = b'\x10l\xe1~\x0f\xda\x08(\xac<\xf9IH(Rn'
    
    # algorithm = algorithms.AES128(long_term_key)
    algorithm = algorithms.AES(long_term_key)
    cipher = Cipher(algorithm, mode=modes.ECB())
    encryptor = cipher.encryptor()
    ct = encryptor.update(randomness)
    return ct

    
def encrypt_cha(bytes,key,nonce):
    chacha = ChaCha20Poly1305(key)
    ct = chacha.encrypt(nonce, bytes)
    #print(len(ct))
    return ct
    
def decrypt_cha(ct,key,nonce):
    chacha = ChaCha20Poly1305(key)
    result = chacha.decrypt(nonce, ct)
    return result


