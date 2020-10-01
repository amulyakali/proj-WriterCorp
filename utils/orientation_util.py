import numpy as np
from scipy.ndimage import interpolation as inter

'''
skew correction code
source : https://towardsdatascience.com/pre-processing-in-ocr-fc231c6035a7
'''


def find_score(arr, angle):
    # data = imutils.rotate_bound(arr, angle)
    # data = cv2.threshold(data,80,255,cv2.THRESH_OTSU)[1]
    # data = data/255
    data = inter.rotate(arr, angle,reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score


def orient(bin_img):
    #cv2.imwrite("bin_img.png",bin_img)
    h = bin_img.shape[0]
    h_10 = int(0.1*h)
    bin_img = bin_img[h_10:(h-h_10),:]
    bin_img = bin_img/255
    delta = 1
    limit = 15
    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        # print("angle",angle,"score",score)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    # cv2.imshow("w1",bin_img)
    # cv2.imshow("w2",imutils.rotate_bound(bin_img,-1*best_angle))
    # cv2.waitKey(0)
    print('Best angle: ',best_angle)
    return -1*best_angle
