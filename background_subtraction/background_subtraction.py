# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:33:55 2020

@author: Clara Brechtel 
"""
import cv2 as cv
import numpy as np

def main():
    part_1()
    part_2()
    part_3()
    bonus()

# Create grayscale image     
def part_1():
    img_0 = cv.imread(r'frames/000000.jpg')
    cv.imshow('Image 0', img_0)
    print(img_0.shape)          # (240, 320, 3)
    print(img_0)
    img_0_gray = cv.cvtColor(img_0, cv.COLOR_BGR2GRAY)
    cv.imshow('Grayscale: Image 0', img_0_gray)
    cv.waitKey(0)
    cv.imwrite(r'output/gray_000000.png', img_0_gray)
    
# Display video and find background image 
def part_2():
    vid = cv.VideoCapture('frames/%06d.jpg')
    gray = []                   
    
    # Load each frame of the video 
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:            # Breaks out of loop if frame can't be read
            break
        
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray.append(gray_frame)
        cv.imshow('Video', gray_frame)
        cv.waitKey(45)          
    vid.release()
    
    # Compute background image with grayscale frames 
    gray = np.array(gray)
    avg_frame = np.mean(gray, axis=0)
    avg_frame = avg_frame.astype(np.uint8)
    cv.imshow('Background Image', avg_frame)
    cv.waitKey(0)
    cv.imwrite(r'output/background_img.png', avg_frame)
    
# Threshold cars in image 
def part_3():
    # Compute the absolute difference between image 0 and the background 
    img_0 = cv.imread(r'output/gray_000000.jpg', cv.IMREAD_GRAYSCALE)
    background = cv.imread(r'output/background_img.png', cv.IMREAD_GRAYSCALE)
    diff = cv.absdiff(img_0, background)
    cv.imshow('Absolute Difference', diff)
    
    # Apply Basic Thresholding
    ret, thresh1 = cv.threshold(diff,40,255,cv.THRESH_BINARY)
    cv.imshow('Basic Thresholding', thresh1)
    
    # Apply Otsu's Thresholding
    ret, thresh2 = cv.threshold(diff,40,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imshow('Otsus Thresholding', thresh2)
    cv.waitKey(0)

# Draw square around each car in video 
def bonus():
    background = cv.imread(r'output/background_img.png', cv.IMREAD_GRAYSCALE)
    vid = cv.VideoCapture('frames/%06d.jpg')
    
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:        
            break
        
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        diff = cv.absdiff(gray_frame, background)
        ret, thresh1 = cv.threshold(diff,40,255,cv.THRESH_BINARY)

        contours,hierarchy = cv.findContours(thresh1, 1, 2)
        color_img = cv.cvtColor(thresh1, cv.COLOR_GRAY2BGR)
        
        for i in range(len(contours)):
            cnt = contours[i]
            x,y,w,h = cv.boundingRect(cnt)
            cv.rectangle(color_img,(x,y),(x+w,y+h),(0,0,255),1)
            
        cv.imshow('Bonus', color_img)
        cv.waitKey(60)
        
    vid.release()
    cv.waitKey(0)
    
if __name__=="__main__":
    main()
    
    
   
    
