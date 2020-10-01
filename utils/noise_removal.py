import cv2
import sys
import imutils
import os
import numpy as np

def removeNoiseAtBounds( image, strip_width=50 ):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 0, 100)
    h, w = canny.shape
    start_y, end_y, start_x, end_x = -1, -1 , -1, -1
    strips = strip_width
    strips_alt = int(strips/2)
    if h > w:
        num = int(h/strips)
        num_alt = int(w/strips_alt)

        for ctr in range(num):
            intensity = np.mean( canny[ ctr*strips: (ctr+1)*strips, : ] )
            print( intensity )
            if start_y == -1 and intensity >= 5: 
                start_y = ctr*strips
                print('Image starts here Y -- ', start_y )
            elif start_y != -1 and intensity >= 5:
                end_y = (ctr+1)*strips
                print('Image ends here Y -- ', end_y )


        for ctr in range(num_alt):
            intensity = ( np.mean( canny[ : , ctr*strips_alt: (ctr+1)*strips_alt ] ) )
            print( intensity )
            if start_x == -1 and intensity >= 5: 
                start_x = ctr*strips_alt
                print('Image starts here X -- ', start_x )
            elif start_x != -1 and intensity >= 5:
                end_x = (ctr+1)*strips_alt
                print('Image ends here X -- ', end_x )

        print( 'Orig img ',h,w,' new img ', canny[ start_y: end_y, start_x: end_x ].shape )
    elif w >= h:
        num = int(w/strips)
        num_alt = int(h/strips_alt)

        for ctr in range(num):
            intensity = ( np.mean( canny[ : , ctr*strips: (ctr+1)*strips ] ) )
            print( intensity )
            if start_x == -1 and intensity >= 5: 
                start_x = ctr*strips
                print('Image starts here X -- ', start_x )
            elif start_x != -1 and intensity >= 5:
                end_x = (ctr+1)*strips
                print('Image ends here X -- ', end_x )


        for ctr in range(num_alt):
            intensity = ( np.mean( canny[ ctr*strips_alt: (ctr+1)*strips_alt , : ] ) )
            print( intensity )
            if start_y == -1 and intensity >= 5: 
                start_y = ctr*strips_alt
                print('Image starts here Y -- ', start_y )
            elif start_y != -1 and intensity >= 5:
                end_y = (ctr+1)*strips_alt
                print('Image ends here Y -- ', end_y )

    if start_x != -1 and end_x != -1 and start_y != -1 and end_y != -1:
        print( 'Reduced all co-ords for ', image.shape , ( start_y, end_y, start_x, end_x ) )
        #cv2.imwrite( 'stripped_results/'+elem , image[ max( start_y - 25, 0 ) : min( end_y + 25, image.shape[0]  ),\
        #        max( start_x - 25, 0 ): min( end_x +25, image.shape[1] ) ] )
        retImage = image[ max( start_y - 25, 0 ) : min( end_y + 25, image.shape[0]  ),\
                max( start_x - 25, 0 ): min( end_x +25, image.shape[1] ) ]
    elif start_x != -1 and end_x != -1 and ( start_y == -1 or end_y == -1 ):
        print( 'Reduced ONLy X co-ords for ', image.shape, ( start_y, end_y, start_x, end_x ) )
        #cv2.imwrite( 'stripped_results/'+elem ,image[ : , max( start_x - 25, 0 ): min( end_x +25, image.shape[1] ) ] )
        retImage = image[ : ,\
                max( start_x - 25, 0 ): min( end_x +25, image.shape[1] ) ]
    elif start_y != -1 and end_y != -1 and ( start_x == -1 or end_x == -1 ):
        print( 'Reduced ONLy Y co-ords for ',  image.shape, ( start_y, end_y, start_x, end_x ) )
        #cv2.imwrite( 'stripped_results/'+elem , image[ max( start_y - 25, 0 ) : min( end_y + 25, image.shape[0]  ) , :  ] )
        retImage = image[ max( start_y - 25, 0 ) : min( end_y + 25, image.shape[0]  ),\
                : ]
    else:
        print('IMAGE COULD NOT BE REDUCED - ',  )
        #cv2.imwrite( 'stripped_results/'+elem , image )
        retImage = image

    return retImage

#cv2.imwrite( '_trial.jpg' , removeNoiseAtBounds( cv2.imread( sys.argv[1] ), int( sys.argv[2] ) ) )
