from __future__ import print_function
import cv2
import numpy as np
from imutils import contours
import imutils
import math
import time


import imutils
from collections import deque
from imutils.video import VideoStream
import argparse

cam = cv2.VideoCapture(2)

cv2.namedWindow("test")

gammaBlue = 3
gammaYellow = 23
gammaOrange = 8
gammaRed = 28
gammaGreen = 0
gammaWhite = 13

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8") 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def test():
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-i", "--image", required=True,
    #	help="path to input image")
    #args = vars(ap.parse_args())
     
    # load the original image
    #original = cv2.imread(args["image"])
    original = cv2.imread('C:\\Users\\User\\Desktop\\test.bmp')

    for gamma in np.arange(0.0, 3.5, 0.5):
	# ignore when gamma is 1 (there will be no change to the image)
        if gamma == 1:
            continue

        # apply gamma correction and show the images
        gamma = gamma if gamma > 0 else 0.1
        adjusted = adjust_gamma(original, 2)
        cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
	    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imshow("Images", np.hstack([original, adjusted]))
        cv2.waitKey(0)

def test22method():
    while True:
        ret, image = cam.read()
        original = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(image.shape, dtype=np.uint8)
    
        colors = {
            'r': ([141, 40, 40], [255, 255, 255]), #Red - x
            'b': ([110,50,50], [130,255,255]), #Blue - x
            'y': ([20, 140, 100], [40, 255, 255]), #Yellow 
            'w': ([70, 10, 130], [180, 110, 255]), #White
            'g': ([36,0,0], [86,255,255]), #Green - x
            'o': ([1, 190, 200], [18, 255, 255]) #Orange
            }
    
        # Color threshold to find the squares
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        for color, (lower, upper) in colors.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            color_mask = cv2.inRange(image, lower, upper)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_kernel, iterations=5)
        
            color_mask = cv2.merge([color_mask, color_mask, color_mask])
            mask = np.zeros(image.shape, dtype=np.uint8)
            mask = cv2.bitwise_or(mask, color_mask)
    
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            # Sort all contours from top-to-bottom or bottom-to-top
            if len(cnts) != 0:
                (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
        
            # Take each row of 3 and sort from left-to-right or right-to-left
            cube_rows = []
            row = []
            for (i, c) in enumerate(cnts, 1):
                #row.clear
                row.append(c)
                if i % 3 == 0:  
                    (cnts, _) = contours.sort_contours(row, method="left-to-right")
                    #cube_rows.clear
                    cube_rows.append(cnts)
                    row = []
        
             #Draw text
            number = 0
            for row in cube_rows:
                for c in row:
                    x,y,w,h = cv2.boundingRect(c)
                    print("{} {} {}".format(x,y,color))
                    cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
                    cv2.putText(original, "{}".format(color), (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    number += 1
           
        cv2.imshow('mask', mask)
        cv2.imwrite('mask.png', mask)
        cv2.imshow('Farbig', original)
        cv2.waitKey(1)

def cornerDedection():
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--buffer", type=int, default=50)
    args = vars(ap.parse_args())
    
    greenLower = (29, 86, 6)
    greenUpper = (64, 255, 255)
    pts = deque(maxlen=args["buffer"])
    
    vs = VideoStream(src=1).start()
    time.sleep(2.0)
    
    while 1:
    	frame = vs.read()
    
    	frame = imutils.resize(frame, width=700)
    	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    	mask = cv2.inRange(hsv, greenLower, greenUpper)
    	mask = cv2.erode(mask, None, iterations=2)
    	mask = cv2.dilate(mask, None, iterations=2)
    
    	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    	cnts = imutils.grab_contours(cnts)
    	center = None
    
    	if len(cnts) > 0:
    		c = max(cnts, key=cv2.contourArea)
    		((x, y), radius) = cv2.minEnclosingCircle(c)
    		M = cv2.moments(c)
    		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    
    		if radius > 10:
    			cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
    
    	pts.appendleft(center)
    
    	for i in range(1, len(pts)):
    		if pts[i - 1] is None or pts[i] is None:
    			continue
    
    		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 4.5)
    		cv2.line(frame, pts[i - 1], pts[i], (0, 255, 250), thickness)
    
    	cv2.imshow("Rubik's cube tracking", frame)
    	key = cv2.waitKey(1) & 0xFF
    
    	if key == ord("q"):
    		break
    
    vs.release()
    cv2.destroyAllWindows()

def nothing(x):
    pass

def test3method():
    while True:
        ret, image = cam.read()
        cv2.flip(image, 2)
        original = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(image.shape, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.merge([mask, mask, mask])
        mask1 = cv2.bitwise_or(mask, color_mask)
        cv2.imshow("Mask", mask1)
        cv2.waitKey(1)

def test2method():
    pos = 10
    cv2.namedWindow('Farbig')
    cv2.createTrackbar('gamma', 'Farbig' , 10, 100, nothing)
    while True:
        #time.sleep(3)
        pos = cv2.getTrackbarPos('gamma','Farbig')
        pos = pos / 10
        if pos == 0:
            pos = 1
        ret, image = cam.read()
        cv2.flip(image, 1)
        original = image.copy()
        

        for gamma in np.arange(0.0, 3.5, 0.5):
	    # ignore when gamma is 1 (there will be no change to the image)
            if gamma == 1:
                continue

            # apply gamma correction and show the images
            adjusted = adjust_gamma(original, pos)
        #blurred = cv2.bilateralFilter(adjusted,9,75,75)
        #blurred = cv2.GaussianBlur(adjusted, (11, 11), 0)
        image = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
        #image = original
        cv2.imshow('image', image)
        mask = np.zeros(image.shape, dtype=np.uint8)
        
        #colors = {
        #    'r': ([141, 40, 40], [255, 255, 255]), #Red - x
        #    'b': ([110,50,50], [130,255,255]), #Blue - x
        #    'y': ([20, 100, 100], [30, 255, 255]), #Yellow 
        #    'w': ([70, 10, 130], [180, 110, 255]), #White
        #    'g': ([36,0,0], [86,255,255]), #Green - x
        #    'o': ([5,50,50], [15, 255, 255]) #Orange
        #    }

        #colors = {
        #    'r': ([120,120,140], [180,250,200]), #Red - x
        #    'b': ([100,100,100], [130,255,255]), #Blue - x
        #    'y': ([20, 100, 100], [40, 255, 255]), #Yellow 
        #    'w': ([70, 10, 130], [180, 110, 255]), #White
        #    'g': ([60,110,110], [100,220,250]), #Green - x
        #    'o': ([5, 150, 150], [15, 235, 250]) #Orange
        #    }

        colors ={
                    'w':([0, 0, 116], [180, 57, 255]),
    
                    #'Light-red':([0,38, 56], [10,255,255]),
                    'o':([10, 38, 71], [20, 255, 255]),
                    'y':([18, 28, 20], [33, 255, 255]),
                    'g':([36, 10, 33], [88, 255, 255]), 
                    'b':([87,32, 17], [120, 255, 255]),
                    #'purple':([138, 66, 39], [155, 255, 255]),
                    'r':([170,112, 45], [180,255,255]),
    
                    #'black':([0, 0, 0], [179, 255, 50]),      
                    } 
    
        # Color threshold to find the squares
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        #cv2.imshow('maskFarbig', mask)
        for color, (lower, upper) in colors.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            color_mask = cv2.inRange(image, lower, upper)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        
            color_mask = cv2.merge([color_mask, color_mask, color_mask])
            mask = np.zeros(image.shape, dtype=np.uint8)
            mask = cv2.bitwise_or(mask, color_mask)
            print(color)
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            dilate = cv2.dilate(gray, open_kernel, iterations=1);
            cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            # Sort all contours from top-to-bottom or bottom-to-top
            contoursSquare = []
            if len(cnts) != 0:
                (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
                cntsList = list(cnts)
                squares = 0;
                for c in cnts:
                    mom = cv2.moments(c)
                    area = cv2.contourArea(c)
                    arch = cv2.arcLength(c, True)
                    squareness = 4 * math.pi * area / math.pow(arch,2)
                    #print(squareness)
                    if squareness >= 0.65 and squareness <= 0.85 and area > 3000 and area < 10000:
                        contoursSquare.append(c)
                        squares+=1
                #for t in elementsToRemove:
                #    cntsList.remove(t)

            # Take each row of 3 and sort from left-to-right or right-to-left
            cube_rows = []
            row = []
            for (i, c) in enumerate(contoursSquare, 1):
                #row.clear
                row.append(c)

                #if i % 3 == 0:  
                (contoursSquare, _) = contours.sort_contours(row, method="left-to-right")
                #cube_rows.clear
                cube_rows.append(row)
                row = []
        
             #Draw text
            number = 0
            for row in cube_rows:
                for c in row:
                    x,y,w,h = cv2.boundingRect(c)
                    cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
                    cv2.putText(original, "{}".format(color), (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    number += 1
        cv2.imshow('mask', mask)
        cv2.imwrite('mask.png', mask)
        cv2.imshow('Farbig', original)

        cv2.waitKey(1)

def getColorFront():
    while True:
        cubeletsPositions = []
        image = cv2.imread('C:\\Users\\User\\Desktop\\test.bmp')
        original = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(image.shape, dtype=np.uint8)
        
        colors = {
            'r': ([141, 40, 40], [255, 255, 255]), #Red - x
            'b': ([110,50,50], [130,255,255]), #Blue - x
            'y': ([20, 140, 100], [40, 255, 255]), #Yellow 
            'w': ([100, 100, 200], [255,255,255]), #White
            'g': ([36,0,0], [86,255,255]), #Green - x
            'o': ([1, 190, 200], [18, 255, 255]) #Orange
            }
        
        # Color threshold to find the squares
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        for color, (lower, upper) in colors.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            color_mask = cv2.inRange(image, lower, upper)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_kernel, iterations=5)
            
            color_mask = cv2.merge([color_mask, color_mask, color_mask])
            mask = np.zeros(image.shape, dtype=np.uint8)
            mask = cv2.bitwise_or(mask, color_mask)
        
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            # Sort all contours from top-to-bottom or bottom-to-top
            if len(cnts) != 0:
                (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
            
            # Take each row of 3 and sort from left-to-right or right-to-left
            cube_rows = []
            row = []
            for (i, c) in enumerate(cnts, 1):
                row.clear
                row.append(c)
                if i % 3 == 0:  
                    (cnts, _) = contours.sort_contours(row, method="left-to-right")
                    cube_rows.clear
                    cube_rows.append(cnts)
                    row = []
            
            # Draw text
            number = 0
            for row in cube_rows:
                for c in row:
                    x,y,w,h = cv2.boundingRect(c)
                    cubelet = [x, y, color]
                    cubeletsPositions.append(cubelet)
                    #print("{} {} {}".format(x,y,color))
                    cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
                    cv2.putText(original, "{}".format(color), (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    number += 1
        return cubeletsPositions


def getTestColorFront():
    while True:
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("test")
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        cubeletsPositions = []
        image = cv2.imread('C:\\Users\\User\\Desktop\\test.bmp')
        original = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(image.shape, dtype=np.uint8)
        
        colors = {
            'b': ([192, 255, 255], [253, 255, 255]), #Blue
            'r': ([0, 255 ,255], [28, 255, 255]), #Red
            'y': ([15, 0, 0], [36, 255, 255]) #Yellow
            }
        
        # Color threshold to find the squares
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        for color, (lower, upper) in colors.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            color_mask = cv2.inRange(image, lower, upper)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_kernel, iterations=5)
            
            color_mask = cv2.merge([color_mask, color_mask, color_mask])
            mask = np.zeros(image.shape, dtype=np.uint8)
            mask = cv2.bitwise_or(mask, color_mask)
        
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            # Sort all contours from top-to-bottom or bottom-to-top
            if len(cnts) != 0:
                (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
            
            # Take each row of 3 and sort from left-to-right or right-to-left
            cube_rows = []
            row = []
            for (i, c) in enumerate(cnts, 1):
                row.clear
                row.append(c)
                if i % 3 == 0:  
                    (cnts, _) = contours.sort_contours(row, method="left-to-right")
                    cube_rows.clear
                    cube_rows.append(cnts)
                    row = []
            
            # Draw text
            number = 0
            for row in cube_rows:
                for c in row:
                    x,y,w,h = cv2.boundingRect(c)
                    cubelet = [x, y, color]
                    cubeletsPositions.append(cubelet)
                    #print("{} {} {}".format(x,y,color))
                    cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
                    cv2.putText(original, "{}".format(color), (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    number += 1
        cv2.imshow('mask', mask)
        cv2.imwrite('mask.png', mask)
        cv2.imshow('original', original)
        cv2.waitKey()
        #return cubeletsPositions
        #return 'wowgybwyogygybyoggrowbrgywrborwggybrbwororbwborgowryby'


































while True:
    ret, frame = cam.read()
    #cv2.imshow("test", frame)
    #cornerDedection()
    test2method()
    #test()
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    if k%256 == 8:
        test22method()
    #if k%256 == 13:
    #    #enter pressed
        
    elif k%256 == 32:
        # SPACE pressed
        while True:
            ret, frame = cam.read()
            cv2.imwrite("C:\\Users\\User\\Desktop\\test.bmp", frame)
            image = cv2.imread('C:\\Users\\User\\Desktop\\test.bmp')
            cv2.flip(image, 1)
            original = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = np.zeros(image.shape, dtype=np.uint8)
        
            colors = {
                'blue': ([69, 120, 100], [179, 255, 255]), #Blue
                'red': ([155, 25, 0], [179, 255, 255]), #Red
                'yellow': ([15, 0, 0], [36, 255, 255]) #Yellow
                }
        
            # Color threshold to find the squares
            open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            for color, (lower, upper) in colors.items():
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                color_mask = cv2.inRange(image, lower, upper)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_kernel, iterations=5)
        
                color_mask = cv2.merge([color_mask, color_mask, color_mask])
                mask = cv2.bitwise_or(mask, color_mask)
        
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            # Sort all contours from top-to-bottom or bottom-to-top
            (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
        
            # Take each row of 3 and sort from left-to-right or right-to-left
            cube_rows = []
            row = []
            for (i, c) in enumerate(cnts, 1):
                row.append(c)
                if i % 3 == 0:  
                    (cnts, _) = contours.sort_contours(row, method="left-to-right")
                    cube_rows.append(cnts)
                    row = []
        
            # Draw text
            number = 0
            for row in cube_rows:
                for c in row:
                    x,y,w,h = cv2.boundingRect(c)
                    cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
                    cv2.putText(original, "#{}".format(number + 1), (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    number += 1
        
            #cv2.imshow('mask', mask)
            cv2.imwrite('mask.png', mask)
            cv2.imshow('original', original)

            cv2.waitKey(0)
       
cam.release()
cv2.destroyAllWindows()