import cv2
import os
import numpy as np
import imutils
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
import json
from utils.contour_classify import get_contour_type
from utils.orientation_util import orient
import time

def smooth(x, window_len=11, window='hanning'):

    """
    Taken from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    # if x.ndim != 1:
    #     raise ValueError, "smooth only accepts 1 dimension arrays."
    #
    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."
    #
    # if window_len < 3:
    #     return x
    #
    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

def removeBorder( bw ):
    '''

    :param bw: binarized image
    :return: remove any horizontal and vertical lines
    '''

    horizontal = np.copy(bw)
    vertical = np.copy(bw)
    #cv.imwrite( fname+'_bw_img.jpg', bw )

    rows, cols = vertical.shape[0], horizontal.shape[1]
    verticalsize, horizontalsize = 50, 50
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    #cv.imwrite('prayer_vertical.jpg', vertical )

    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    #cv.imwrite('prayer_hori.jpg', horizontal )

    minus_vertical = cv2.subtract( bw, vertical )
    minus_horizontal = cv2.subtract( minus_vertical, horizontal )
    #cv.imwrite( fname+'_subtracted.jpg', minus_horizontal )
    return minus_horizontal
#print( removeBorder( cv.imread( sys.argv[1] ), sys.argv[1] ) )

# Function to generate horizontal projection profile
def getHorizontalProjectionProfile(image):
    # Convert white spots to ones
    image[image == 255] = 1

    horizontal_projection = np.sum(image, axis=1)

    return horizontal_projection

def transform_coords(modified_roi_lines,y_min,x_min):
    res = []
    for line in modified_roi_lines:
        # line = [l["pts"] for l in line]
        modified_line = []
        for roi in line:
            id = roi["id"]
            [x,y,w,h] = roi["pts"]
            x = x_min + x
            y = y_min + y
            modified_line.append({"id":id,"pts":[x,y,w,h]})
        res.append(modified_line)
    return res

def is_separated_by_vertical_line(connecting_line,vertical_lines):
    for ver_line in vertical_lines:
        line = LineString(connecting_line)
        other = LineString(ver_line)
        if line.intersects(other):
            return line.intersection(other)
    return False

def contains_same_heights_2(line):
    line = sorted(line,key=lambda x:x[3])
    first_height = line[0][3]
    last_height = line[-1][3]
    same_heights = True
    max_diff = max(3,0.2 * min(first_height, last_height))
    # max_diff = 0.1 * min(first_height, last_height)
    if abs(first_height - last_height) > max_diff:
        # and not (curr_height < 0.6*prev_height and (abs(prev_ymin-curr_ymin) < 0.75*prev_height)):
        same_heights = False
    return same_heights

def contains_same_heights(line):
    same_heights = True
    prev_height = line[0][3]
    prev_ymin = line[0][1]
    for block in line:
        curr_height = block[3]
        curr_ymin = block[1]
        # max_diff = 0.1 * min(curr_height, prev_height)
        max_diff = max(3, 0.2 * min(curr_height, prev_height))
        if abs(curr_height-prev_height) > max_diff :
                # and not (curr_height < 0.6*prev_height and (abs(prev_ymin-curr_ymin) < 0.75*prev_height)):
            same_heights = False
            break
        # if not (curr_height < 0.6*prev_height and (abs(prev_ymin-curr_ymin) < 0.75*prev_height)):
        prev_height = block[3]
        prev_ymin = block[1]
    return same_heights

def combine(merge_words):
    x_mins = []
    y_mins = []
    x_maxs = []
    y_maxs = []
    text = ""
    word_ids = []
    for word in merge_words:
        [x_min,y_min,w,h] = word['pts']
        x_max = x_min+w
        y_max = y_min+h
        x_mins.append(x_min)
        y_mins.append(y_min)
        x_maxs.append(x_max)
        y_maxs.append(y_max)
        text = text + " " + word["text"]
        word_ids.append(word["id"])
    return {"text":text.strip(),"pts":[min(x_mins),min(y_mins),max(x_maxs),max(y_maxs)],"ids":word_ids}

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

class TextDetection:
    '''
    class to identify blobs of text in an image
    '''
    def __init__(self,img,path,cls_model,debug=False):
        # self.path = path
        self.debug = debug
        self.basename = os.path.basename(path)
        self.cls_model = cls_model
        self.modified_path = os.path.join("mod_jpg/", os.path.basename(path))

        self.img = img
        self.im_shape = self.img.shape
        threshold_img = self.threshold(self.img)
        # cv2.imwrite("oriented/"+self.basename,orient(threshold_img))
        self.alignment = orient(threshold_img)
        # self.alignment = -3

        ## align all the thresholded images and the original image
        self.oriented_img_thresh = imutils.rotate_bound(threshold_img, self.alignment)
        self.oriented_orig_img = imutils.rotate_bound(self.img, self.alignment)

        # imutils rotate_bound is messing up the thresholding, so applying it again
        self.oriented_img_thresh = cv2.threshold(self.oriented_img_thresh, 80, 255, cv2.THRESH_OTSU)[1]
        cv2.imwrite(self.modified_path,self.oriented_orig_img)
        self.contours_info = {}
        self.current_contour_idx = 0

        cv2.imwrite("res/" + "thresh_" + self.basename, threshold_img)
        self.regions = self.get_roi(self.oriented_img_thresh)
        self.lines = self.sort_contours(self.regions)
        self.common_height = self.get_common_height()
        self.lines = self.remove_noise()
        self.lines = self.verifyLines()

    def check_if_line(self,line):
        max_w = max([r["pts"][2] for r in line])
        if (len(line) > 0 or (max([l["pts"][2] for l in line]) > 30 and max([l["pts"][3] for l in line]) > 15)) :
            return True
        return False

    def is_new_line(self,curr_coords, curr_line):
        [curr_x,curr_y,curr_w,curr_h] = curr_coords
        [prev_x,prev_y,prev_w,prev_h] = curr_line[-1]['pts']
        curr_ymax = curr_y + curr_h
        prev_ymax = prev_y + prev_h
        if abs(curr_y-prev_y)>0.75*curr_h or abs(curr_ymax-prev_ymax)>0.75*curr_h :
            return True
        if len(curr_line) > 1 and abs(curr_y-prev_y)>3:
            ## check if the sequence of increasing/decreasing x co-ords has changed
            [prev2_x, prev2_y, prev2_w, prev2_h] = curr_line[-2]['pts']
            if (curr_x - prev_x) * (prev_x-prev2_x)<0:
                return True
        return False

    def sort_contours(self,rois):
        '''

        :param rois: extracted rois
        :return: sorted lines top to bottom and words left to right in each line
        '''
        rois = sorted(rois, key=lambda x: x["pts"][1])
        lines = []
        line = []
        if len(rois) > 1:
            line.append(rois[0])
            curr_ymin = rois[0]["pts"][1]
            curr_h = rois[0]["pts"][3]
            curr_ymax = curr_ymin + curr_h
            for c in rois[1:]:
                [xmin, ymin, width, height] = c["pts"]
                ymax = ymin + height
                if self.is_new_line(c["pts"],line):
                    if len(line) > 0 and self.check_if_line(line):
                        lines.append(line)
                    line = []
                    line.append(c)
                else:
                    # (height < 0.6*curr_h and (abs(ymin - curr_ymin) < 0.75 * curr_h) ): ## to include comma in the same line
                    line.append(c)
            if len(line) > 0 and self.check_if_line(line):
                lines.append(line)
            # sort contours inside the line
        for idx, line in enumerate(lines):
            sorted_line = sorted(line, key=lambda x: x["pts"][0])
            lines[idx] = sorted_line
        return lines

    def get_common_height(self):
        heights = []
        for line in self.lines:
            for word in line:
                heights.append(word["pts"][3])
        if len(heights)>0:
            return np.bincount(heights).argmax()
        else:
            return 0

    def resize_image(self,img):
        final_width = 2000
        fx = final_width / img.shape[1]
        fy = fx
        if final_width < img.shape[1]:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
        resized = cv2.resize(img, None, fx=fx, fy=fy, interpolation=interpolation)
        print("final size", resized.shape)
        return resized

    def remove_bg_old(self,image,blur_window=51):
        [h, w] = image.shape[:2]
        if blur_window!=51:
            blur_window = int(min(h / 4, w / 4))
            if blur_window % 2 == 0:
                blur_window += 1
        blur_bg = cv2.medianBlur(image, blur_window)

        # if self.debug:
        cv2.imwrite("debug/blur_bg.jpg", np.bitwise_not(blur_bg))
        # wo_bg = cv2.subtract(np.bitwise_not(image), np.bitwise_not(blur_bg))
        wo_bg = np.subtract(np.bitwise_not(image).astype(int), np.bitwise_not(blur_bg).astype(int))
        wo_bg[wo_bg < -50] = 255
        wo_bg[wo_bg < 0] = 0
        wo_bg[wo_bg > 25] = 255
        wo_bg = wo_bg.astype(np.uint8)
        # if self.debug:
        cv2.imwrite("debug/wo_bg.jpg", wo_bg)
        morphKernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        wo_bg = cv2.morphologyEx(wo_bg, cv2.MORPH_CLOSE, morphKernel)
        return wo_bg

    def threshold(self,image):
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
            if not (len(c) < 5 or h < 4 or h>0.1*orig.shape[0]):
                cv2.drawContours(mask, [c], -1, 255, -1)

            # else:
                # print("w", w, "h", h)
                # cv2.rectangle(orig, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # mask = cv2.bitwise_not(mask)
        cleaned = cv2.bitwise_and(cleaned, mask)
        dil_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        clean_dilate = cv2.dilate(cleaned, dil_ker)
        contours = cv2.findContours(clean_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        to_classify = []
        to_classify_coords = []
        for c in contours:
            [x, y, w, h] = cv2.boundingRect(c)
            if not (len(c) < 5 or h <= 5):
                to_classify.append(im_out[y:y + h, x:x + w,:])
                to_classify_coords.append(c)
        preds = get_contour_type(to_classify,self.cls_model)
        irr_mask = np.zeros(cleaned.shape[:2], np.uint8)
        for idx, pred in enumerate(preds):
            if pred[0] < 0.4:
                c = to_classify_coords[idx]
                cv2.drawContours(irr_mask, [c], -1, 255, -1)

        irr_mask = cv2.bitwise_not(irr_mask)
        cv2.imwrite("irr.jpg",irr_mask)
        cleaned_fin = cv2.bitwise_and(cleaned, irr_mask)
        cleaned_fin = removeBorder(cleaned_fin)
        return cleaned_fin

    def getLines(self,img,thresh_fin_img, h, w):
        # cv2.imwrite("line_ip_thresh.jpg",thresh_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 15, -2)
        thresh_img = cv2.bitwise_or(thresh_img,thresh_fin_img)
        cv2.imwrite("adap_thresh.jpg", thresh_img)
        horizontal = np.copy(thresh_img)
        vertical = np.copy(thresh_img)
        rows = h
        cols = w
        verticalsize = rows // 30
        horSize = cols // 10

        print("vertical size", verticalsize)
        print("horSize", horSize)
        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        horStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horSize, 1))

        verticalStructure_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3, verticalsize))
        horStructure_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (horSize, 3))

        # Apply morphology operations
        vertical = cv2.erode(vertical, verticalStructure)
        # cv2.imwrite("res/ver_erode.jpg",vertical)
        vertical = cv2.dilate(vertical, verticalStructure_dil, iterations=3)

        hor = cv2.erode(horizontal, horStructure)
        # cv2.imwrite("res/hor_erode.jpg", hor)
        hor = cv2.dilate(hor, horStructure_dil, iterations=3)
        lines = cv2.bitwise_or(hor, vertical)
        return hor,vertical,lines

    def getVerticalLines(self):
        # print("PATH--", path)
        # im = cv2.imread(self.modified_path)
        im = self.oriented_orig_img
        im_out = im.copy()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                  cv2.THRESH_BINARY, 15, -2)
        vertical = np.copy(bw)
        rows = vertical.shape[0]
        verticalsize = rows // 30
        verticalsize_dil = rows // 20
        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        verticalStructure_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3, verticalsize+150))
        # Apply morphology operations
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)

        ## to remove lines appearing during scanning
        vertical_2 = cv2.dilate(vertical, verticalStructure_dil)
        mask = np.zeros(vertical.shape[:2], np.uint8)
        contours, hierarchy = cv2.findContours(vertical_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for idx,c in enumerate(contours):
            sort_c = sorted(c,key=lambda x:x[0][1])
            pt1 = tuple(sort_c[0][0])
            pt2 = tuple(sort_c[-1][0])
            h = abs(pt1[1]-pt2[1])
            if h > 0.7 * vertical.shape[0]:
                cv2.drawContours(mask, [c], -1, 255, -1)

        cv2.imwrite("vertical_mask.jpg", mask)
        cv2.imwrite("vertical.jpg",vertical)

        ## remove lines which stretch across the entire page
        vertical = cv2.bitwise_and(vertical,cv2.bitwise_not(mask))
        # cv2.imwrite("vertical_new.jpg", vertical_new)
        contours,hierarchy = cv2.findContours(vertical, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        vertical_lines = []
        for idx,c in enumerate(contours):
            sort_c = sorted(c,key=lambda x:x[0][1])
            pt1 = tuple(sort_c[0][0])
            pt2 = tuple(sort_c[-1][0])
            h = abs(pt1[1]-pt2[1])
            if h<0.5*vertical.shape[0]:
                cv2.line(im_out,pt1,pt2,(0,0,255),2)
                vertical_lines.append([pt1, pt2])
        cv2.imwrite("ver_lines.jpg",im_out)
        return vertical_lines

    def get_roi(self,img_thresh):
        cv2.imwrite("thresh.jpg",img_thresh)
        canvas = self.oriented_orig_img.copy()
        # remove all lines
        bw = img_thresh
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        gradX = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, rectKernel)

        thresh = cv2.threshold(gradX, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # if self.debug:
        cv2.imwrite("res/thresh_" + self.basename, thresh)

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        rois = []
        mask = np.zeros(self.oriented_orig_img.shape[:2], np.uint8)
        for idx,c in enumerate(cnts):
            [x, y, w, h] = cv2.boundingRect(c)
            # define main island contour approx. and hull
            epsilon = 0.05 * cv2.arcLength(c, True)
            if len(c) > 5 and (h>10 or (h>5 and w*h>80)) and w>0:
                # cv2.drawContours(canvas,[c],-1,(0,0,255),2)
                cv2.rectangle(canvas,(x,y),(x+w,y+h),(0,0,255),2)
                self.contours_info[self.current_contour_idx] = c
                rois.append({"id":self.current_contour_idx,"pts":[x,y,w,h]})
                self.current_contour_idx+=1
            else:
                cv2.drawContours(mask, [c], -1, 255, -1)

        # mask = np.zeros(self.oriented_orig_img.shape[:2], np.uint8)
        # cv2.drawContours(mask, cnts,-1, 255, -1)
        # cv2.imwrite("res.jpg",mask)
        cv2.imwrite("mask.jpg",mask)
        cv2.imwrite("canvas.jpg",canvas)
        return rois

    def split_text_by_vertical_lines(self):
        fin_rois = []
        for region in self.regions:
            [x1, y1, w, h] = region["pts"]
            y_mid = int(y1 + (h/2))
            intersecting_pt_top = is_separated_by_vertical_line([(x1, y1), (x1+w, y1)], self.verticalLines)
            intersecting_pt_bottom = is_separated_by_vertical_line([(x1, y1+h), (x1 + w, y1+h)], self.verticalLines)
            if intersecting_pt_top and intersecting_pt_bottom:
                inter_x = int(intersecting_pt_top.x)
                if inter_x-1-x1 > 0:
                    fin_rois.append({"id":str(self.current_contour_idx),"pts":[x1,y1,inter_x-1-x1,h]})
                    self.contours_info[str(self.current_contour_idx)] = self.contours_info[region["id"]]
                    self.current_contour_idx += 1
                if int(w-(inter_x+2-x1)) > 0:
                    fin_rois.append({"id":str(self.current_contour_idx),"pts":[inter_x+2,y1,int(w-(inter_x+2-x1)),h]})
                    self.contours_info[str(self.current_contour_idx)] = self.contours_info[region["id"]]
                    self.current_contour_idx += 1
            else:
                fin_rois.append(region)
        return fin_rois

    def get_errored_contours(self,line_idx):
        line = self.lines[line_idx]
        perfect_line = False
        to_verify = line
        to_verify = sorted(to_verify, key=lambda x: x["pts"][3]) # sort by height
        res = []
        max_line_height = max([l["pts"][3] for l in line])
        while not perfect_line and len(to_verify)>0 and not \
                ((len(line)>1 and contains_same_heights_2([t["pts"] for t in to_verify])) or (len(line) == 1 and (max_line_height < 1.9 * self.common_height))):
            res.append(to_verify[-1])
            to_verify = to_verify[:-1]
        return res,to_verify

    # def is_steep(self,data,trough_idx):
    #     left_vals = [data[trough_idx-1],data[trough_idx-2],data[trough_idx-3],data[trough_idx-4],data[trough_idx-5]]
    #     right_vals = [data[trough_idx+1],data[trough_idx+2],data[trough_idx+3],data[trough_idx+4],data[trough_idx+5]]
    #     curr_val = data[trough_idx]
    #     left_diff = [abs(v-curr_val) for v in left_vals]
    #     right_diff = [abs(v-curr_val) for v in right_vals]
    #     if max(left_diff)>35 :
    #         return True
    #     return False

    def is_steep(self,data,trough_idx):
        peak_val = data[trough_idx]
        if peak_val>-400:
            break_point = 35
        else:
            break_point = 200
        for idx in reversed(range(0,trough_idx)):
            curr_val = data[idx]
            if peak_val-curr_val>break_point:
                return True
            elif curr_val > peak_val:
                break
        return False


    def is_relevant_minima(self,data,trough_idx):
        height = len(data)
        if height > 0.5*self.common_height:
            if height-10>trough_idx>10 and self.is_steep(data,trough_idx):
                return True
        return False


    def get_contours(self,thresh,canvas,x_start,y_start):

        [height,width] = thresh.shape[:2]
        if int(width/5)>3:
            rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width/5), 1))
            thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,rectKernel)
        hor_projection = getHorizontalProjectionProfile(thresh.copy())
        hor_projection = np.array([-1 * p for p in hor_projection])
        smooth_data = smooth(hor_projection)
        troughs = argrelmax(smooth_data, order=3)
        split_points = [0]
        res_contours = []
        for t in troughs[0]:
            if self.is_relevant_minima(smooth_data,t):
                split_points.append(t)
                if self.debug:
                    plt.plot(t,smooth_data[t],"g*")
        if self.debug:
            plt.plot(smooth_data)
            plt.show()
        for idx,y_coord in enumerate(split_points):
            if y_coord == 0:
                ymin = y_coord
            else:
                ymin = y_coord - 2
            if idx == len(split_points)-1:
                y_max = height
            else:
                y_max = split_points[idx+1]+2
            if y_max - ymin > 10:
                if self.debug:
                    cv2.rectangle(canvas,(0,ymin),(width,y_max),(0,255,0),3)
                snip_thresh = thresh[ymin:y_max,:]
                cnts = cv2.findContours(snip_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
                for c in cnts:
                    [x,y,w,h] = cv2.boundingRect(c)
                    self.current_contour_idx+=1
                    res_contours.append({"id":self.current_contour_idx,"pts":[x_start+x,y_start+ymin+y,w,h]})

        if self.debug:
            cv2.imshow("w2", canvas)
            cv2.waitKey(0)
        return res_contours



    def split_text_lines(self,errored_contours):
        thresh_img = self.oriented_img_thresh
        res_contours = []
        for c in errored_contours:
            [x,y,w,h] = c["pts"]
            if h>2*self.common_height:
                canvas = self.oriented_orig_img.copy()
                mask = np.zeros(canvas.shape[:2],dtype=np.uint8)
                contour = self.contours_info[c["id"]]
                cv2.drawContours(mask,[contour],-1,255,-1)
                canvas = np.bitwise_and(thresh_img,mask)
                bw = canvas
                rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 1))
                gradX = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, rectKernel)

                thresh = cv2.threshold(gradX, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                res_contours.extend(self.get_contours(thresh[y:y+h,x:x+w],self.oriented_orig_img.copy()[y:y+h,x:x+w,:],x,y))
            else:
                res_contours.append(c)
        return res_contours


    def extract_roi_using_sp(self,line_indices,snip_thresh_dilated):
        # hor_projection = getHorizontalProjectionProfile(
        #     cv2.bitwise_and(snip_thresh_dilated, cv2.bitwise_not(snip_lines_img)))
        # hor_projection = np.array([-1 * p for p in hor_projection])
        # smooth_data = smooth(hor_projection)
        # # troughs = find_peaks(hor_projection)
        # window = signal.general_gaussian(5, p=0.5, sig=10)
        # filtered = signal.fftconvolve(window, hor_projection)
        # filtered = (np.average(hor_projection) / np.average(filtered)) * filtered
        # # smooth_data = np.roll(filtered, -25)
        # troughs_2 = argrelmax(smooth_data, order=3)
        # # plt.plot(hor_projection)
        # # smooth_data = pd.Series(hor_projection).rolling(window=5).mean().plot(style='k')
        # # smooth_data = savgol_filter(hor_projection, 11, 3)
        # # plt.plot(smooth_data)
        # # for t in troughs_2[0]:
        # #     if smooth_data[t] > -400:
        # #         plt.plot(t,smooth_data[t],"g*")
        # # plt.show()
        # # return False
        rois = []
        for idx in line_indices:
            line = [l["pts"] for l in self.lines[idx]]
            max_line_height = max([l[3] for l in line])
            if (len(line)>1 and contains_same_heights(line)) or (len(line) == 1 and (max_line_height < 1.9 * self.common_height)):
                rois.extend(self.lines[idx])
            else:
                errored_contours, correct_contours = self.get_errored_contours(idx)
                rois.extend(correct_contours)
                rois.extend(self.split_text_lines(errored_contours))
        return rois

    def extract_cropped_text(self, y_min, y_max, x_min, x_max, line_indices_to_modify):
        snippet_color = self.oriented_orig_img[y_min:y_max, x_min:x_max]
        cp = snippet_color.copy()
        # cv2.imshow("w",cp)
        # cv2.waitKey(0)
        # snippet = cv2.cvtColor(self.oriented_orig_img[y_min:y_max, x_min:x_max],cv2.COLOR_BGR2GRAY)
        # snip_thresh = cv2.threshold(snippet, 80, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
        snip_thresh = self.oriented_img_thresh[y_min:y_max, x_min:x_max]
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        snip_dilated = cv2.morphologyEx(snip_thresh, cv2.MORPH_CLOSE, rectKernel)

        rectKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        snip_dilated_2 = cv2.morphologyEx(snip_thresh, cv2.MORPH_OPEN, rectKernel2)
        snip_thresh_dilated = cv2.threshold(snip_dilated, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # for idx in line_indices_to_modify:
        #     for word in self.lines[idx]:
        #         [x,y,w,h] = word["pts"]
        #         word_snip = self.oriented_orig_img[y:y+h, x:x+w]
        #         word_snip = cv2.cvtColor(word_snip, cv2.COLOR_BGR2GRAY)
        #         word_snip_thresh = cv2.threshold(word_snip, 80, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
        #         word_snip_dilated = cv2.morphologyEx(word_snip_thresh, cv2.MORPH_CLOSE, rectKernel)
        #         word_snip_thresh_dilated = cv2.threshold(word_snip_dilated, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #         word_snip_thresh_dilated = cv2.bitwise_and(word_snip_thresh_dilated,
        #                                               cv2.bitwise_not(self.all_lines_img[y:y+h, x:x+w]))
        #         word_hor_projection = getHorizontalProjectionProfile(
        #             cv2.bitwise_and(word_snip_thresh_dilated, cv2.bitwise_not(self.all_lines_img[y:y+h, x:x+w])))
        #         word_hor_projection = [-1 * p for p in word_hor_projection]
        #         cv2.imshow("w",word_snip_thresh)
        #         cv2.waitKey(0)
        #         smooth_data = smooth(word_hor_projection)
        #         troughs_2 = argrelmax(smooth_data, order=3)
        #         # plt.plot(hor_projection)
        #         # smooth_data = pd.Series(hor_projection).rolling(window=5).mean().plot(style='k')
        #         # smooth_data = savgol_filter(hor_projection, 11, 3)
        #         plt.plot(smooth_data)
        #         for t in troughs_2[0]:
        #             if smooth_data[t] > -400:
        #                 plt.plot(t, smooth_data[t], "g*")
        #         plt.show()

        contours = cv2.findContours(snip_thresh_dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        rois = []
        for idx,c in enumerate(contours):
            [x,y,w,h] = cv2.boundingRect(c)
            epsilon = 0.1 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(c) > 5 and h > 8:
                rois.append({"id":"c_"+str(idx),"pts":[x,y,w,h]})
                self.contours_info["c_"+str(idx)] = c
                cv2.drawContours(cp,c,-1,(0,0,255),-1)
                # cv2.rectangle(cp,(x,y),(x+w,y+h),(0,0,255),3)
        roi_lines = self.sort_contours(rois)
        is_perfect = True
        for line in roi_lines:
            line = [l["pts"] for l in line]
            max_line_height = max(l[3] for l in line)
            if ((not contains_same_heights(line)) or (len(line)==1 and (max_line_height > 1.9*self.common_height))):
                is_perfect = False
                break
        if is_perfect:
            return transform_coords(roi_lines, y_min, x_min)
        else:
            rois = self.extract_roi_using_sp(line_indices_to_modify,snip_thresh_dilated)
            return self.sort_contours(rois)

    def remove_noise(self):
        fin_lines = []
        for line_idx,line in enumerate(self.lines):
            max_w = max([l["pts"][2] for l in line])
            max_h = max([l["pts"][3] for l in line])
            if len(line)>1:
                min_w = 2*self.common_height
            else:
                min_w = 2*self.common_height
            if (max_w>min_w and max_h > 0.5*self.common_height) or max_h>=0.75*self.common_height:
                fin_lines.append(line)
        return fin_lines

    def verifyLines(self):
        fin_lines = []
        modified_indices = []
        for line_idx,line in enumerate(self.lines):
            line = [l["pts"] for l in line]
            if line_idx not in modified_indices :
                max_line_height =  max([r[3] for r in line])
                if (not contains_same_heights(line)) or (max_line_height > 1.9*self.common_height):
                    y_max = max([b[1]+b[3] for b in line])
                    line_indices_to_modify = []
                    for idx in range(line_idx,len(self.lines)):
                        curr_line = self.lines[idx]
                        curr_line = [c["pts"] for c in curr_line]
                        curr_max_line_height = max([r[3] for r in curr_line])
                        curr_y2_max = max([b[1] + b[3] for b in curr_line])
                        if contains_same_heights(curr_line) and (curr_max_line_height <= 1.9*self.common_height) :
                            curr_y2_min = min([b[1] + b[3] for b in curr_line])
                            if curr_y2_max <= y_max:
                                line_indices_to_modify.append(idx)
                                y_max = max([b[1] + b[3] for b in curr_line])
                            # elif y_max-curr_y2_min>-2:
                            #     line_indices_to_modify.append(idx)
                            #     y_max = curr_y2_max
                            else:
                                break
                        elif (curr_y2_max - y_max)>2*self.common_height :
                            break
                        else:
                            line_indices_to_modify.append(idx)
                            y_max = max([b[1] + b[3] for b in curr_line])
                    if len(line_indices_to_modify) > 0:
                        x_min = line[0][0]
                        x_max = line[0][0] + line[0][2]
                        y_min = line[0][1]
                        y_max = line[0][1]+line[0][3]
                        for idx in line_indices_to_modify:
                            curr_line = self.lines[idx]
                            curr_line = [l["pts"] for l in curr_line]
                            line_x_min = min(b[0] for b in curr_line)
                            line_x_max = max(b[0]+b[2] for b in curr_line)
                            line_y_min = min(b[1] for b in curr_line)
                            line_y_max = max(b[1]+b[3] for b in curr_line)
                            if line_x_min < x_min:
                                x_min = line_x_min
                            if line_x_max > x_max:
                                x_max = line_x_max
                            if line_y_min < y_min:
                                y_min = line_y_min
                            if line_y_max > y_max:
                                y_max = line_y_max
                        # cv2.imshow("w",self.oriented_orig_img[y_min:y_max,x_min:x_max])
                        # cv2.waitKey(0)
                        modified_roi_lines = self.extract_cropped_text(y_min, y_max, x_min, x_max, line_indices_to_modify)
                        if modified_roi_lines:
                            fin_lines.extend(modified_roi_lines)
                            modified_indices.extend(line_indices_to_modify)
                        else:
                            fin_lines.append(self.lines[line_idx])
                else:
                    fin_lines.append(self.lines[line_idx])
        return fin_lines

    def checkIfVerLinePasses(self,sent_obj):
        [x1,y1,x2,y2] = sent_obj['pts']
        y_mid = int(y1 + ((y2-y1)/2))
        intersecting_pt = is_separated_by_vertical_line([(x1,y_mid),(x2,y_mid)],self.verticalLines)
        if intersecting_pt:
            xMaxArr = []
            xMinArr = []
            yMinArr = []
            yMaxArr = []
            idArr = []
            word_ids = sent_obj["ids"]
            for id in word_ids:
                wrd = self.words[id]
                for char_text,char_vertices in zip(wrd['texts'],wrd["vertices"]):
                    xAll = list()
                    yAll = list()
                    for vertex in char_vertices:
                        xAll.append(vertex["x"])
                        yAll.append(vertex["y"])
                    xMaxArr.append(max(xAll))
                    xMinArr.append(min(xAll))
                    yMinArr.append(min(yAll))
                    yMaxArr.append(max(yAll))
                    idArr.append(id)
            # print(xMaxArr)
            # print(xMinArr)
            char_idx = 0
            char_xmins  = list()
            char_xmaxs = list()
            char_ymins = list()
            char_ymaxs = list()
            for char in list(sent_obj['text']):
                if char!= " ":
                    char_xmins.append(xMinArr[char_idx])
                    char_xmaxs.append(xMaxArr[char_idx])
                    char_ymins.append(yMinArr[char_idx])
                    char_ymaxs.append(yMaxArr[char_idx])
                    char_idx+=1
                else:
                    char_xmins.append(None)
                    char_xmaxs.append(None)
                    char_ymins.append(None)
                    char_ymaxs.append(None)
            intersecting_x = intersecting_pt.x
            split_idx = 0
            for idx,char_xmax in enumerate(char_xmaxs):
                if char_xmax and char_xmax > intersecting_x:
                    split_idx = idx
                    break
            if split_idx > 0:
                 res = []
                 chars_list = list(sent_obj['text'])
                 prev_idx = None
                 for i in range(split_idx-1,-1,-1):
                    if char_xmaxs[i]:
                        prev_idx = i
                        break
                 if prev_idx:
                     res.append({"text":"".join(chars_list[:split_idx]),"pts":[x1,y1,char_xmaxs[prev_idx],char_ymaxs[prev_idx]],\
                                 "ids":list(set(idArr[:split_idx]))})
                     res.append({"text": "".join(chars_list[split_idx:]),
                                 "pts": [char_xmins[split_idx], char_ymins[split_idx],x2,y2], \
                                 "ids": list(set(idArr[split_idx:]))})
                     return res
                 else:
                     return [sent_obj]
            else:
                return [sent_obj]
        else:
            return [sent_obj]

    def merge_close_texts(self):
        im_width = self.im_shape[1]
        if im_width<1500:
            max_inter_word_spacing = 20
            min_inter_word_spacing = 2
            min_word_height = 5
        else:
            max_inter_word_spacing = int(15*(im_width/1500))
            min_inter_word_spacing = int(2*(im_width/1500))
            min_word_height = int(5*(im_width/1500))
        fin_lines = []

        # merge blocks if the gap is less than max_inter_word_spacing
        for l in self.lines:
            line_words = l
            # line_words = [word for word in line_words if (word['pts'][3])>min_word_height and len(word['text'])>0]
            # if sum([c.isalpha() for c in l['text']]) > 0 or sum([c.isdigit() for c in l['text']]) > 0:
            if len(line_words) > 1:
                split_indices = []
                prev_x_max = line_words[0]["pts"][0]+line_words[0]["pts"][2]
                prev_char_height = line_words[0]["pts"][3]

                for idx, block in enumerate(line_words[1:]):
                    [x1, y1, w, h] = block["pts"]
                    x2 = x1+w
                    y2 = y1+h
                    curr_char_height = h
                    max_inter_word_spacing = int(2 * max(prev_char_height,curr_char_height)) ## calculate the spacing allowed based on the character width
                    curr_x_min = x1
                    curr_y_mid = int(y1 + ((y2 - y1) / 2))
                    spacing = curr_x_min - prev_x_max
                    # if "Survey Nos" in l["text"]:
                    #     print(spacing)
                    # save the indices where distance is large enough to be considered as separate blocks
                    # and merge the rest

                    # the contours split using vertical lines share one side and the length of connecting line will be 0
                    # so add some more length
                    if prev_x_max+3 == curr_x_min:
                        connecting_line = [[min(curr_x_min+2,im_width),curr_y_mid],[max(0,prev_x_max-2),curr_y_mid]]
                    else:
                        connecting_line = [[curr_x_min,curr_y_mid],[prev_x_max,curr_y_mid]]

                    if spacing > max_inter_word_spacing:
                        split_indices.append(idx + 1)
                    # elif spacing > min_inter_word_spacing:
                    #     if is_separated_by_vertical_line([(prev_x_max, curr_y_mid), (curr_x_min, curr_y_mid)],
                    #                                      self.verticalLines):
                    #         split_indices.append(idx + 1)
                    prev_x_max = x2
                    prev_char_height = curr_char_height
                if len(split_indices) > 0:
                    prev_idx = 0
                    line_blocks = []
                    for idx in split_indices:
                        combined = combine(line_words[prev_idx:idx])
                        line_blocks.append(combined)
                        prev_idx = idx
                    combined = combine(line_words[prev_idx:])
                    line_blocks.append(combined)
                    fin_lines.append(line_blocks)
                else:
                    fin_lines.append([combine(l)])
            elif len(line_words)==1:
                fin_lines.append([combine(line_words[0:1])])
        return fin_lines

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)



