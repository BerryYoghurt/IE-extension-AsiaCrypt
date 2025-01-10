# IE-extension

To encrypt an image, run `python3 -m image_encryption -o <output_filename> <input_filename>` in the parent directory of `image_encryption` package.

To set up the decryption environment, get the server running by `python3 encryption_server.py`. The server needs to be up and running for the decryption to work.
Then, import the extension (IE-encryption-frontend) to Chrome. For more instructions on how to import an extension, see https://developer.chrome.com/docs/extensions/get-started/tutorial/hello-world#load-unpacked.

To use the decryption environment, upload an encrypted image to a website. Then, open the image (so that it has the same original size), right click on it, then choose "Decrypt this image".

## If you are on a Unix-like system, you will get a slightly higher quality if you do the following:
First go to jpeg-9f and start 'make'. For information on compiling jpeg-9f, see install.txt.

After compiling, move 'djpeg' to IE-encryption-backend and rename it as 'mydjpeg'.
Then, go to IE-encryption-backend/image_encryption/jpeg.py, search for "UNIX-LIKE". You need to uncomment two lines and comment out one line. Sorry for the inconvinience!

## If you are on Windows:
Compiling jpeg-9f is painful, so it is better to use the all-Python code.

