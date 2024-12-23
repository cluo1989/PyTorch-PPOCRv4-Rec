import cv2
import math
import numpy as np


def resize_norm_img(img, image_shape):
    imgH, imgW, imgC = image_shape

    h, w = img.shape[:2]
    ratio = w / float(h)

    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))

    # resize
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')

    # normalize
    resized_image = resized_image.astype('float32')
    resized_image = resized_image / 255
    resized_image -= 0.5
    resized_image /= 0.5

    # padding
    padding_im = np.zeros((imgH, imgW, imgC), dtype=np.float32)
    padding_im[:, :resized_w, ...] = np.expand_dims(resized_image, axis=-1)
    return padding_im


def random_pad(img):
    # 0.1% randomly pad black border
    if np.random.uniform(0, 1) <= 0.001:
        ph = 0
        pw = np.random.randint(32)
        pv = 0  # pad black

        hwc = list(img.shape)
        hwc[0] += 2*ph
        hwc[1] += 2*pw

        if pv is None:
            pv = 0
        elif pv == -1:
            pv = img.mean()

        pad_img = pv * np.ones(hwc, dtype=img.dtype)
        pad_img[ph:hwc[0]-ph, pw:hwc[1]-pw, ...] = img
        img = pad_img

    return img
