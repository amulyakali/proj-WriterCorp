from __future__ import division
from __future__ import print_function
from keras.preprocessing.image import apply_affine_transform,apply_brightness_shift
from keras.preprocessing.image import array_to_img, img_to_array
import random
import numpy as np
import cv2
from PIL import ImageEnhance

def apply_contrast_shift(x, contrast):
    if ImageEnhance is None:
        raise ImportError('Using brightness shifts requires PIL. '
                          'Install PIL or Pillow.')
    x = array_to_img(x)
    x = imgenhancer_contrast = ImageEnhance.Contrast(x)
    x = imgenhancer_contrast.enhance(contrast)
    x = img_to_array(x)
    return x

def apply_sharpness_shift(x, sharpness):
    if ImageEnhance is None:
        raise ImportError('Using sharpness shifts requires PIL. '
                          'Install PIL or Pillow.')
    x = array_to_img(x)
    imgenhancer_sharp = ImageEnhance.Sharpness(x)
    x = imgenhancer_sharp.enhance(sharpness)
    x = img_to_array(x)
    return x


def preprocess_old(img, imgSize, dataAugmentation=False):
    "put img into target img of size imgSize, transpose for TF and normalize gray-values"

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])

    # increase dataset size by applying random stretches to the images
    if dataAugmentation:
        img = img.reshape(img.shape[0],img.shape[1],1)
        brightness = np.random.uniform(0.5, 1.5)
        contrast = np.random.uniform(0.5, 1.5)
        rotate = np.random.uniform(-4, 4)
        shear = np.random.uniform(-2, 2)
        sharp = np.random.uniform(0.2, 2)
        image_affine = apply_affine_transform(img, theta=rotate, shear=shear)
        image_bright = apply_brightness_shift(image_affine, brightness=brightness)
        image_contrast = apply_contrast_shift(image_bright, contrast=contrast)
        image_sharp = apply_sharpness_shift(image_contrast, sharpness=sharp)
        stretch = (random.random() - 0.5)  # -0.5 .. +0.5
        wStretched = max(int(img.shape[1] * (1 + stretch)), 1)  # random width, but at least 1
        img = cv2.resize(image_sharp, (wStretched, image_sharp.shape[0]))

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    # f = fy
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img
    cv2.imwrite("16_8_before_transpose.jpg",target)

    # transpose for TF
    img = cv2.transpose(target)
    cv2.imwrite("16_8_after_transpose.jpg",img)


    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img
    cv2.imwrite("10_8_norm.jpg",img)
    return img

def preprocess(img, imgSize, dataAugmentation=False):
    "put img into target img of size imgSize, transpose for TF and normalize gray-values"

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])

    # increase dataset size by applying random stretches to the images
    if dataAugmentation:
        img = img.reshape(img.shape[0],img.shape[1],1)
        brightness = np.random.uniform(0.5, 1.5)
        contrast = np.random.uniform(0.5, 1.5)
        rotate = np.random.uniform(-4, 4)
        shear = np.random.uniform(-2, 2)
        sharp = np.random.uniform(0.2, 2)
        image_affine = apply_affine_transform(img, theta=rotate, shear=shear)
        image_bright = apply_brightness_shift(image_affine, brightness=brightness)
        image_contrast = apply_contrast_shift(image_bright, contrast=contrast)
        image_sharp = apply_sharpness_shift(image_contrast, sharpness=sharp)
        stretch = (random.random() - 0.5)  # -0.5 .. +0.5
        wStretched = max(int(img.shape[1] * (1 + stretch)), 1)  # random width, but at least 1
        img = cv2.resize(image_sharp, (wStretched, image_sharp.shape[0]))

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    target = np.ones([ht, wt]) * 255
    if h<ht and w<wt:
        top_padding = int((ht-h)/2)
        left_padding = int((wt-w)/2)
        target[top_padding:top_padding+h, left_padding:left_padding+w] = img
    else:
        fx = w / wt
        fy = h / ht
        f = max(fx, fy)
        newSize = (max(min(wt, int(w / f)), 1),
                   max(min(ht, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
        img = cv2.resize(img, newSize)
        top_padding = int((ht-newSize[1])/2)
        left_padding = int((wt-newSize[0])/2)
        target[top_padding:top_padding+newSize[1], left_padding:left_padding+newSize[0]] = img
    # cv2.imwrite("16_8_before_transpose.jpg", target)

    # transpose for TF
    img = cv2.transpose(target)
    # cv2.imwrite("16_8_after_transpose.jpg", img)

    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img
    # cv2.imwrite("16_8_norm.jpg", img)
    return img

def preprocess_2(img, imgSize, dataAugmentation=False):
    "put img into target img of size imgSize, transpose for TF and normalize gray-values"

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])

    # increase dataset size by applying random stretches to the images
    if dataAugmentation:
        img = img.reshape(img.shape[0],img.shape[1],1)
        brightness = np.random.uniform(0.5, 1.5)
        contrast = np.random.uniform(0.5, 1.5)
        rotate = np.random.uniform(-4, 4)
        shear = np.random.uniform(-2, 2)
        sharp = np.random.uniform(0.2, 2)
        image_affine = apply_affine_transform(img, theta=rotate, shear=shear)
        image_bright = apply_brightness_shift(image_affine, brightness=brightness)
        image_contrast = apply_contrast_shift(image_bright, contrast=contrast)
        image_sharp = apply_sharpness_shift(image_contrast, sharpness=sharp)
        stretch = (random.random() - 0.5)  # -0.5 .. +0.5
        wStretched = max(int(img.shape[1] * (1 + stretch)), 1)  # random width, but at least 1
        img = cv2.resize(image_sharp, (wStretched, image_sharp.shape[0]))

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    target = np.ones([ht, wt]) * 255
    if h<ht and w<wt:
        top_padding = 0
        left_padding = 0
        target[top_padding:top_padding+h, left_padding:left_padding+w] = img
    else:
        fx = w / wt
        fy = h / ht
        f = max(fx, fy)
        newSize = (max(min(wt, int(w / f)), 1),
                   max(min(ht, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
        img = cv2.resize(img, newSize)
        top_padding = 0
        left_padding = 0
        target[top_padding:top_padding+newSize[1], left_padding:left_padding+newSize[0]] = img
    # cv2.imwrite("16_8_before_transpose.jpg", target)

    # transpose for TF
    img = cv2.transpose(target)
    # cv2.imwrite("16_8_after_transpose.jpg", img)

    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img
    # cv2.imwrite("16_8_norm.jpg", img)
    return img
