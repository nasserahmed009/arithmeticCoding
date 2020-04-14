## How to use
---
Make sure that you have `python` installed on your device, `openvc` and `numpy`. Then open the folder [arithmeticCodingNoScaling or arithmeticCodingWithScaling] in your terminal and type ..
### Encoding
`python encoder.py` 
You will be asked to enter the image file name, the required block size and the dataType just in case of arithmeticCodingNoScaling folder is used.

This script will generate files that the decoder will use it later in order to be able to decode the image.

### Decoding
for decoding the image just run the decoding script by typing ..
`python decoder.py`
the script will start decoding showing a progress bar untill the end of the decoding process and tha output image will be saved as **output.jpg**