# IE-extension

## `pip` requirements
1. Install `IE-extension-backend/requirements.txt`
2. Our protocol uses a computer-vision model, CLIP, which needs to be installed depeding on the OS you are running on. You will need to install `pytorch` and CLIP. See https://github.com/openai/CLIP?tab=readme-ov-file#usage.

## Usage
To encrypt an image, run `python3 -m image_encryption -o <output_filename> <input_filename>` in the parent directory of `image_encryption` package.

To set up the decryption environment, get the server running by `python3 encryption_server.py`. The server needs to be up and running for the decryption to work.
Then, import the extension (IE-encryption-frontend) to Chrome. For more instructions on how to import an extension, see https://developer.chrome.com/docs/extensions/get-started/tutorial/hello-world#load-unpacked.

To use the decryption environment, upload an encrypted image to a website. Then, open the image (so that it has the same original size), right click on it, then choose "Decrypt this image".

## If you are on a UNIX-like system, you will get a slightly higher quality if you do the following:
1. Go to IE-extension-backend/jpeg-9f.
2. Check install.txt to see how to install it for your system. Note that we apply a small modification on the original jpeg-9f library, so we need this local compilation step.

Generally, on Mac, you will need to rename some files and run `make`. On Linux, you will need to first run `./configure` and **then** run `make`.

## If you are on Windows or otherwise face trouble successfully compiling jpeg-9f:
The code has an all-Python alternative. Just delete the directory `IE-extension-backend/jpeg-9f`.

