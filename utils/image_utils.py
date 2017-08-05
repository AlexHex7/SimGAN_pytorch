import numpy as np


def normalize_img(img):
    # numpy
    img = img * 1.0 / 255
    return (img - 0.5) / 0.5
    # return img


def restore_img(img):
    img += max(-img.min(), 0)
    if img.max() != 0:
        img /= img.max()
    img *= 255
    img = img.astype(np.uint8)
    return img