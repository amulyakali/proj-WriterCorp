import cv2
import numpy as np

def resize_image(img):
    h = 50
    w = 170
    return cv2.resize(img, (w, h))

def get_contour_type(contours,model):
    X = []
    for c in contours:
        X.append(resize_image(c))
    X = np.array(X)
    X = X / 255.0
    preds = model.predict(X)
    return preds



