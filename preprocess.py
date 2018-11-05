import numpy as np
import cv2
import math

def compute_skew(src):
    
    #load in grayscale:
    #src = cv2.imread(file_name,0)
    height, width = src.shape[0:2]
    
    #invert the colors of our image:
    #cv2.bitwise_not(src, src)
    
    #Hough transform:
    minLineLength = width/2.0
    maxLineGap = 20
    lines = cv2.HoughLinesP(src,1,np.pi/180,100,minLineLength,maxLineGap)
    
    #calculate the angle between each line and the horizontal line:
    angle = 0.0
    nb_lines = len(lines)
    
    
    for line in lines:
        angle += math.atan2(line[0][3]*1.0 - line[0][1]*1.0,line[0][2]*1.0 - line[0][0]*1.0);
    
    angle /= nb_lines*1.0
    
    return angle* 180.0 / np.pi


def deskew_(img):
    angle = compute_skew(img)
    #load in grayscale:
    #img = cv2.imread(file_name,0)
    
    #invert the colors of our image:
    #cv2.bitwise_not(img, img)
    
    #compute the minimum bounding box:
    non_zero_pixels = cv2.findNonZero(img)
    center, wh, theta = cv2.minAreaRect(non_zero_pixels)
    
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rows, cols = img.shape
    rotated = cv2.warpAffine(img, rot_mat, (cols, rows), flags=cv2.INTER_CUBIC)


    #Border removing:
    sizex = np.int0(wh[0])
    sizey = np.int0(wh[1])
    print(theta)
    if theta > -45 :
        temp = sizex
        sizex= sizey
        sizey= temp
    #return cv2.getRectSubPix(rotated, (sizey,sizex), center)
    return rotated


def deskew(img):
    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    print('Angle before correction' + str(angle))
    
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle
       

    print('Angle after correction' + str(angle))
    
    # rotate the image to deskew it
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
       
    return rotated

'''
def deskew(imgfile):
    from math import radians, degrees

    from pdftabextract.common import ROTATION, SKEW_X, SKEW_Y
    from pdftabextract.geom import pt
    from pdftabextract.textboxes import rotate_textboxes, deskew_textboxes
    from pdftabextract import imgproc
    iproc_obj = imgproc.ImageProc(imgfile)

    # find rotation or skew
    # the parameters are:
    # 1. the minimum threshold in radians for a rotation to be counted as such
    # 2. the maximum threshold for the difference between horizontal and vertical line rotation (to detect skew)
    # 3. an optional threshold to filter out "stray" lines whose angle is too far apart from the median angle of
    #    all other lines that go in the same direction (no effect here)
    rot_or_skew_type, rot_or_skew_radians = iproc_obj.find_rotation_or_skew(radians(0.5),    # uses "lines_hough"
                                                                            radians(1),
                                                                            omit_on_rot_thresh=radians(0.5))

    # rotate back or deskew text boxes
    needs_fix = True
    if rot_or_skew_type == ROTATION:
        print("> rotating back by %f°" % -degrees(rot_or_skew_radians))
        rotate_textboxes(p, -rot_or_skew_radians, pt(0, 0))
    elif rot_or_skew_type in (SKEW_X, SKEW_Y):
        print("> deskewing in direction '%s' by %f°" % (rot_or_skew_type, -degrees(rot_or_skew_radians)))
        deskew_textboxes(p, -rot_or_skew_radians, rot_or_skew_type, pt(0, 0))
    else:
        needs_fix = False
        print("> no page rotation / skew found")

    return deskew_image
'''
def preprocess(imgfile, remove_lines=False):
    image = cv2.imread(imgfile)
    
    # Non-local Means (NLM)
    denoised = cv2.fastNlMeansDenoising(image, None,h=10,templateWindowSize=7,searchWindowSize=21) 
    
    # Transform source image to gray if it is not already
    if len(denoised.shape) != 2:
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    else:
        gray = denoised
        
    gray = cv2.bitwise_not(gray)
    
   
    # Deskewing
    rotated = deskew(gray)
    
    # Thresholding
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # Otsu
    #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)  # Adaptive  
        
    # Deskewing
    #rotated = deskew(thresh)
    #rotated = deskew(imgfile)
    clean_image = cv2.bitwise_not(thresh)
    
    # Lines removal
    if remove_lines:
        # Create the images that will use to extract the horizontal and vertical lines
        horizontal = np.copy(rotated)
        # Specify size on horizontal axis
        cols = horizontal.shape[1]
        horizontal_size = int(cols / 30)
        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        # Apply morphology operations
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)
        #inverse the image, so that lines are black for masking
        horizontal_inv = cv2.bitwise_not(horizontal)
        #perform bitwise_and to mask the lines with provided mask
        print(clean_image.shape)
        print(horizontal_inv.shape)
        
        masked_img = cv2.bitwise_and(rotated, rotated, mask=horizontal_inv)
        #reverse the image back to normal
        masked_img_inv = cv2.bitwise_not(masked_img)
        clean_image = masked_img_inv
        
    return clean_image