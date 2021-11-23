import numpy as np
import cv2

def read_bytecode(img_add):
    bytecode = open(img_add, 'r')
    bytecode = [line for line in bytecode]
    bytecode = bytecode[0].split('\t')[:-1]
    return bytecode

def bytecode_to_image(bytecode, w1):
    gray_image = []
    for c in bytecode:
        gray_image.append(int(c, 16))
    gray_image = np.array(gray_image)

    if len(bytecode) % w1 != 0:
        gray_image = np.concatenate([gray_image, np.zeros(w1 - (len(bytecode) % w1))])

    gray_image = gray_image.reshape(int(gray_image.shape[0] / w1), w1)

    return gray_image.astype('uint8')

def scale_image(image, w1, h2):
    try:
        scaled = cv2.resize(image, (w1, h2), interpolation = cv2.INTER_AREA)
        return scaled
    except Exception as e:
        print(str(e), image.shape, image)

def dir_transformation(img_dir, w1, h2):
    images = []
    for img_add in img_dir:
        x = read_bytecode(img_add)
        if len(x) == 0:
            continue
        if len(x) < w1:
            print(len(x))
            continue
        x = bytecode_to_image(x, w1)
        x = scale_image(x, w1, h2)
        x = x / 255
        x = np.expand_dims(x, axis = -1)
        images.append(x)
    return images

def encoder_dir_transformation(img_dir, w1):
    images = []
    for img_add in img_dir:
        x = read_bytecode(img_add)
        if len(x) == 0:
            continue
        if len(x) < w1:
            print(len(x))
            continue
        x = bytecode_to_image(x, w1)
        x = x / 255
        x = np.expand_dims(x, axis = -1)
        images.append(x)
    return images