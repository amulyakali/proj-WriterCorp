import cv2
import numpy as np
import os

dst_folder = "contours"
counter = 1
prefix = "im644_"

def remove_bg(img):
    '''

    :param img:
    :return: remove all the background colors, shadows, watermarks..
    '''
    # Convert the image to grayscale
    gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gr = cv2.medianBlur(gr,3)

    # Make a copy of the grayscale image
    bg = gr.copy()

    # Apply morphological transformations
    for i in range(5):
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (2 * i + 1, 2 * i + 1))
        bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, kernel2)
        bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel2)


    bg = cv2.medianBlur(bg,5)
    # Subtract the grayscale image from its processed copy
    dif = cv2.subtract(bg, gr)

    # Apply thresholding
    # bw = cv2.threshold(dif, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    bw = cv2.adaptiveThreshold(dif, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, -2)
    dark = cv2.adaptiveThreshold(bg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, -2 )
    # dark = cv2.threshold(bg, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cv2.imshow("bw",bw)
    # cv2.imshow("dark",dark)
    # Extract pixels in the dark region
    # darkpix = gr[np.where(dark > 0)]

    # Threshold the dark region to get the darker pixels inside it
    # darkpix = cv2.threshold(darkpix, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Paste the extracted darker pixels in the watermark region
    # bw[np.where(dark > 0)] = darkpix.T
    # cv2.imshow("bw2",bw)
    # cv2.waitKey(0)
    return bw

def remove_snp(img):
    img = cv2.bitwise_not(img)
    median_blur = cv2.medianBlur(img, 3)
    return cv2.bitwise_and(median_blur,img)

def threshold(image):
    orig = image.copy()
    im_out = image.copy()
    rm = remove_bg(image)
    cleaned = remove_snp(rm)
    # cleaned = cv2.bitwise_not(cleaned)
    cl_orig = cleaned.copy()
    contours = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    mask = np.zeros(cl_orig.shape[:2], np.uint8)

    for c in contours:
        [x, y, w, h] = cv2.boundingRect(c)
        if not (len(c) < 5 or h < 4 or h > 0.1 * orig.shape[0]):
            cv2.drawContours(mask, [c], -1, 255, -1)

        # else:
        # print("w", w, "h", h)
        # cv2.rectangle(orig, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # mask = cv2.bitwise_not(mask)
    cleaned = cv2.bitwise_and(cleaned, mask)
    dil_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    clean_dilate = cv2.dilate(cleaned, dil_ker)
    contours = cv2.findContours(clean_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    for c in contours:
        [x, y, w, h] = cv2.boundingRect(c)
        if not (len(c) < 5 or h <= 5):
            cv2.imwrite(os.path.join(dst_folder,prefix+str(counter)+".jpg"),im_out[y:y + h, x:x + w])
            counter+=1

in_folder = "samples-class-2"
for im in os.listdir(in_folder):



