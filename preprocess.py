import numpy as np
import cv2

def preprocess(image, remove_lines=False):

    # Non-local Means (NLM)
    denoised = cv2.fastNlMeansDenoising(image, None,h=10,templateWindowSize=7,searchWindowSize=21) 
    
    # Transform source image to gray if it is not already
    if len(denoised.shape) != 2:
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    else:
        gray = denoised
        
    gray = cv2.bitwise_not(gray)
    
   
    
    # Thresholding
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # Otsu
    #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)  # Adaptive  
        
    # Deskewing
    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

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


    # rotate the image to deskew it
    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
       
    clean_image = cv2.bitwise_not(rotated)
    
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