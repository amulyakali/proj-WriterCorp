#Document image orientation correction
#This approach is based on text orientation

#Assumption: Document image contains all text in same orientation

import cv2, os
import numpy as np
import utils.noise_removal as NR

debug = True
arr_ = []

#rotate the image with given theta value
def rotate(img, theta):
    rows, cols = img.shape[0], img.shape[1]
    image_center = (cols/2, rows/2)

    M = cv2.getRotationMatrix2D(image_center,theta,1)

    abs_cos = abs(M[0,0])
    abs_sin = abs(M[0,1])

    bound_w = int(rows * abs_sin + cols * abs_cos)
    bound_h = int(rows * abs_cos + cols * abs_sin)

    M[0, 2] += bound_w/2 - image_center[0]
    M[1, 2] += bound_h/2 - image_center[1]

    # rotate orignal image to show transformation
    rotated = cv2.warpAffine(img,M,(bound_w,bound_h),borderValue=(255,255,255))
    return rotated


def slope(x1, y1, x2, y2):
    if x1 == x2:
        return 0
    slope = (y2-y1)/(x2-x1)
    theta = np.rad2deg(np.arctan(slope))
    return theta

def snippet( img ):

    snip = max( 200, min( int( img.shape[0]/2 ), int( img.shape[1]/2 ) ) )
    #snip = 200
    center_h, center_w = int( img.shape[0]/2 ), int( img.shape[1]/2 )
    return img[ center_h - snip: center_h + snip, center_w - snip : center_w + snip, :]

def deskew(filePath):
    '''
    input arg : abs file path of image
    RETURNS : dictionary { 'uncropped':< orig img with correct orientation >, 
                           'uncropped_flip': < above image flipped by 180 >,
                           'cropped': < CROPPED img with correct orientation >,
                           'cropped_flip': < cropped image flipped by 180 >,
                          } 
    '''
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print('begin loop for ', filePath )
    img = cv2.imread(filePath)
    chunk_size = 600
    scale_factor = 1.2
    mode = ''
    print('SHAPE = ',img.shape )
    if img.shape[0] >= img.shape[1]:
        ht_, wd_ = 2800, 2000
        mode = 'ht_'
    elif img.shape[0] < img.shape[1]:
        ht_, wd_ = 2000, 2800
        mode = 'wd_'
    
    ## original and cropped images for final rotation
    cropped_img = NR.removeNoiseAtBounds( img )
    original_img = img.copy()
    ## original and cropped images for final rotation

    img = cv2.resize( cropped_img.copy(), ( wd_, ht_ ) )
    print( img.shape )
    img = snippet( img )
    print( img.shape )
    fullSz = img.shape
    textImg = img.copy()
    arr_ = []
    angle_D = dict()

    small = cv2.cvtColor(textImg, cv2.COLOR_BGR2GRAY)

    #find the gradient map
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    #dilation = cv2.dilate(small,kernel,iterations = 1)
    #display(dilation, fname+'_1')
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    #grad1 = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
    #display(grad1, fname+'_1')

    #Binarize the gradient image
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite('bw.jpg', bw )
    #display(bw, fname+'_2')

    #kernal value (9,1) can be changed to improved the text detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    #display(connected, fname+'_3')

    # using RETR_EXTERNAL instead of RETR_CCOMP
    #contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, dtype=np.uint8)
    #display(mask)
    #cumulative theta value
    cummTheta = 0
    #number of detected text regions
    ct = 0
    x_arr, y_arr = [], []
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        #fill the contour
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        #display(mask, fname+'_3.1')
        #ratio of non-zero pixels in the filled region
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
        #assume at least 45% of the area is filled if it contains text
        if r > 0.45 :#and ( ( mode == 'ht_' and h >= 50 and h <= 70 and h < w*0.6 ) or ( mode == 'wd_' and w >= 50 and w <= 70 and w < h*0.6 ) ):
        #if r > 0.45 and w > 10 and h > 10 and ( h > 2*w or w > 2*h   ):
            #cv2.rectangle(textImg, (x1, y), (x+w-1, y+h-1), (0, 255, 0), 2)

            rect = cv2.minAreaRect(contours[idx])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(textImg,[box],0,(0,0,255),2)
            x_arr.append( box[0][0] )
            y_arr.append( box[0][1] )
            #we can filter theta as outlier based on other theta values
            #this will help in excluding the rare text region with different orientation from ususla value 
            theta = slope(box[0][0], box[0][1], box[1][0], box[1][1])
            #print('Gnarly Theta = ', theta, ' WD and HT ', w, h )
            #if abs(theta) > 0:
            if 1 == 1:
                cummTheta += theta
                ct +=1 
                arr_.append( int(theta) )
                if int(theta) in angle_D.keys():
                    angle_D[ int(theta) ] += 1
                else:
                    angle_D[ int(theta) ] = 1
            #print("Theta", theta)
    
    #find the average of all cumulative theta value
    if ct == 0: 
        orientation = 0
        top_, bottom_ = 0, 0
    else:
        #orientation = cummTheta/ct
        if 1 == 1:
            arr_.sort()
            maxxer, max_an = -1, -10000
            for ct_ in range( int( len(arr_)/2 ) ):
                an_ = arr_[ ct_ ]
                if angle_D[ an_ ] > maxxer:
                    maxxer = angle_D[ an_ ]
                    max_an = an_

            orientation = max_an
            #orientation = arr_[ int(len(arr_)/2) - 1 ]
            print( 'median = ARR median - ', orientation, max_an )

        if mode == 'ht_' and ( orientation >= 0 and orientation <= 45 ):
            orientation = orientation
            print( ' median = ', orientation, "mode == 'ht_' and h < w*0.5 - ORIG OR 0 and 45", h, w  )
        elif mode == 'ht_' and ( orientation > 45 and orientation <= 90 ):
            orientation =  -1* (90 - orientation)
            print( ' median = ', orientation, "mode == 'ht_' and h < w*0.5 - ORIG OR 45 and 90", h, w  )
        elif mode == 'wd_' and ( orientation >= 0 and orientation <= 45 ):
            orientation = ( 90 + orientation )
            print( ' median = ', orientation, "mode == 'wd_' and w < h*0.5 - ORIG OR 0 and 45", h, w  )
        elif mode == 'wd_' and ( orientation > 45 and orientation <= 90 ):
            orientation = ( 180 - orientation )
            print( ' median = ', orientation, "mode == 'wd_' and w < h*0.5 - ORIG OR 45 and 90", h, w  )
   
    ## signature block needs to be chunk size x 3 ..our chunk size is going to be 400 x 400
    ## one at the center and 4 at each corner of the center 
    bw_center_h, bw_center_w = int(bw.shape[0]/2), int(bw.shape[1]/2)
    block_sz = chunk_size*3
    signature_block = bw[ bw_center_h - int( block_sz/2 ): bw_center_h + int( block_sz/2 ), \
                          bw_center_w - int( block_sz/2 ): bw_center_w + int( block_sz/2 ) ]
    print( 'Size of signature block - ',signature_block.shape )
    block1 = np.mean( signature_block[ : chunk_size, : chunk_size ] ) ## 0-400 on both x and y axis
    block2 = np.mean( signature_block[ : chunk_size, chunk_size*2 : ] ) ## 0-400 on y byt 800-1200 on x
    block_center = np.mean( signature_block[ chunk_size : 2*chunk_size, : chunk_size : 2*chunk_size ] ) # 400-800
    block3 = np.mean( signature_block[ 2*chunk_size: , : chunk_size ] ) ## 0-400 on x and 800-1200 y axis
    block4 = np.mean( signature_block[ 2*chunk_size: , 2*chunk_size : ] ) ## 800-1200 on y  x
    print('Signature of this image = ', block1, block2, block_center, block3, block4 )

    if mode == 'wd_':
        print('WDtop 25% pixel density - ', np.mean( bw[ int(0.2*bw.shape[0]): int(0.4*bw.shape[0]), : ] ) )
        print('WDlast 25% pixel density - ', np.mean( bw[ int(0.6*bw.shape[0]): int(0.8*bw.shape[0]), : ] ) )
        top_    = block1 + block2
        bottom_ = block3 + block4 
        print('TOP n BOTT = ', top_, bottom_ )
        if (scale_factor)*top_ < bottom_ :
            print('Adding 180 more since angle might not be suff')
            orientation += 180
        print("Image orientation in degress: ", orientation)
    elif mode == 'ht_':
        print('LEFT 25% pixel density - ', np.mean( bw[ : , int(0.2*bw.shape[1]) : int(0.4*bw.shape[1]) ] ) )
        print('RIGHT 25% pixel density - ', np.mean( bw[ : ,int(0.6*bw.shape[1]) : int(0.8*bw.shape[1]) ] ) )
        left_ = block1 + block3
        right_= block2 + block4
        print('LEFT n RIGHT = ', left_, right_ )
        if (scale_factor)*left_ < right_ :
            print('Adding 180 more since angle might not be suff')
            orientation += 180
        print("Image orientation in degress: ", orientation)

    finalImage = rotate( original_img, orientation)
    cropImage  = rotate( cropped_img, orientation)

    dd_ = { 'uncropped': finalImage,\
                           'uncropped_flip': rotate( finalImage, 180),\
                           'cropped': cropImage,\
                           'cropped_flip': rotate( cropImage, 180)\
                          }
    return dd_
    
if __name__ == "__main__":
    folder = "samples-a"
    for im in os.listdir(folder):
        im_p = os.path.join(folder,im)
        res = deskew(im_p)
        cv2.imwrite(os.path.join("deskew",im),res["cropped"])