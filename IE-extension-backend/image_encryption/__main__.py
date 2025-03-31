from image_encryption.jpeg import encrypt, gen_keys
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
    parser.add_argument("-p", "--password", help="Password. If not given, a new password is generated.")
    parser.add_argument("-o", "--output", help="Output filename.", default="ctxt.jpeg")
    args = parser.parse_args()

    if args.password == None:
        alphabet = string.ascii_letters + string.digits
        password = None
        while True:
            password = ''.join(secrets.choice(alphabet) for i in range(10))
            if (any(c.islower() for c in password)
                    and any(c.isupper() for c in password)
                    and sum(c.isdigit() for c in password) >= 3):
                break
        print("The new password is: {password}")
    else:
        password = args.password

    image = Image.open(args.filename)
    key, mac_key = gen_keys(password)
    # mac_key = b'\x19V\xcc\xe3\xc8\xd6\xa2h\x82\x97\xc7\x9e\x83\x04\x15s\xfd\x06"<7\xa0US\xf1\xe9\xbf\xe1\x9eM\xbe\x94'
    encrypt(95, args.filename, args.output, key, mac_key)

    
