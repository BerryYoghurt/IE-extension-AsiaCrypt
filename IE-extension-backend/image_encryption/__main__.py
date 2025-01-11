from image_encryption.jpeg import encrypt
import string
import secrets
import argparse
from PIL import Image
import math

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="image_encryption",
        description="JPEG-tolerant image encryption")
    parser.add_argument("filename")
    parser.add_argument("-o", "--output", help="Output filename.", default="ctxt.jpeg")
    args = parser.parse_args()

    image = Image.open(args.filename)
    # w16, h16 = math.ceil(image.size[0] / 16) * 16, math.ceil(image.size[1] / 16) * 16
    # key, mac_key = gen_keys(w16, h16, password)
    mac_key = b'\x19V\xcc\xe3\xc8\xd6\xa2h\x82\x97\xc7\x9e\x83\x04\x15s\xfd\x06"<7\xa0US\xf1\xe9\xbf\xe1\x9eM\xbe\x94'
    encrypt(70, args.filename, args.output, mac_key)

    
