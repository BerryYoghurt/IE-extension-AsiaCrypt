# IE-extension
### Contents
1. [Requirements](#requirements)
2. [Usage](#usage)
3. [Sample run](#sample-run)

## Requirements
- Python 3.10 or higher https://www.python.org/downloads/
- Pip is automatically packaged with Python downloaded in the previous step, but in case your Python installation doesn't have it, please follw the instructions in the Pip website https://pip.pypa.io/en/stable/installation/

### `pip` requirements
Run `pip install -r ./IE-extension-backend/requirements.txt`

For completeness, the requirements are:
```
cryptography==44.0.0
numpy==2.2.1
scipy==1.15.1
pillow==11.1.0
jpeglib==1.0.1
requests-toolbelt==1.0.0
setuptools==80.9.0
open-clip-torch==3.2.0
```


### If you are on a UNIX-like system, you will get a slightly higher quality if you do the following:
1. Go to IE-extension-backend/jpeg-9f.
2. Check install.txt to see how to install it for your system. Note that we apply a small modification on the original jpeg-9f library, so we need this local compilation step.

Generally, on Mac, you will need to rename `makefile.xc` to `makefile` and `jconfig.xc` to `jconfig`, and then run `make`. On Linux, you will need to run `./configure` and then run `make`, **without any renaming**.

### If you are on Windows or otherwise face trouble successfully compiling jpeg-9f:
The code has an all-Python alternative. Just delete the directory `IE-extension-backend/jpeg-9f`.

## Usage
To encrypt an image, navigate to the parent directory of `image_encryption` package, e.g. `cd IE-extension-backed` and run `python3 -m image_encryption -o <output_filename> -p <password> <input_filename>`.

To set up the decryption environment:
1. Get the server running by navigating to `IE-extension-backend` and running  `python3 encryption_server.py`. The server needs to be up and running for the decryption to work.
2. Then, import the extension (IE-encryption-frontend) to Chrome. For more instructions on how to import an extension, see https://developer.chrome.com/docs/extensions/get-started/tutorial/hello-world#load-unpacked.
The extension should appear in your list of extensions, e.g.

<kbd><img width="397" height="413" alt="image" src="https://github.com/user-attachments/assets/9df9bc9d-57bd-4aea-9407-5260a9f3736d" /></kbd>.

Then, when you right-click on an image, a context menu item with the extension icon should appear, e.g.

<kbd><img width="293" height="350" alt="dialogue - Copy" src="https://github.com/user-attachments/assets/dc15bab3-bdac-4e09-be2e-8dc7329331f2" /></kbd>.

3. To use the decryption environment, upload an encrypted image to a website. Then, open the image (so that it has the same original size), right click on it, then choose "Decrypt this image". The page might need to be refreshed if this is the first time the extension is used after being loaded to the browser.

## Sample run
1. Encrypting the image:
<kbd><img width="1892" height="218" alt="image" src="https://github.com/user-attachments/assets/610005f7-e1ff-434d-acf1-84566d9c8dca" /></kbd>

Note: this environment uses a virtual environment to install python requirements, but it is optional.

2. Uploading to Facebook:
<kbd><img width="1850" height="984" alt="image" src="https://github.com/user-attachments/assets/59cb66a3-f5a3-42c6-94aa-503d869bd6fa" /></kbd>

3. Decrypting:
<kbd><img width="1845" height="987" alt="image" src="https://github.com/user-attachments/assets/ff0e3f65-585d-4f91-b393-5498b3c264e0" /></kbd>

<kbd><img width="1848" height="980" alt="image" src="https://github.com/user-attachments/assets/0cbfb104-f4f1-41af-8d22-3780477f3a95" /></kbd>

4. Result:
<kbd><img width="1848" height="977" alt="image" src="https://github.com/user-attachments/assets/f678c35d-f12f-4959-a8f8-407392e57750" /></kbd>





