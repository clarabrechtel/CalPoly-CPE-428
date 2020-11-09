# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:51:28 2020

@author: clara
"""

import cv2 as cv 
import numpy as np 
from skimage.util import img_as_float, img_as_uint

def main():
    cat = cv.imread(r'cat.bmp')
    cat = img_as_float(cat)
    cv.imshow('Cat', cat)
    
    dog = cv.imread(r'dog.bmp')
    dog = img_as_float(dog)
    cv.imshow('Dog', dog)
    
    g = cv.getGaussianKernel(31, 5, cv.CV_32F)
    low_pass = g*g.transpose()
    cv.imshow('Low-Pass Kernel', low_pass*255)
    
    all_pass = np.zeros((31, 31), dtype=np.float32)
    all_pass[15, 15] = 1
    high_pass = all_pass - low_pass 
    cv.imshow('High-Pass Kernel', high_pass)

    
    low_dog = cv.filter2D(dog, -1, low_pass)
    cv.imshow('Dog with Low-Pass Filter', low_dog)
    high_cat = cv.filter2D(cat, -1, high_pass)
    cv.imshow('Cat with High-Pass Filter', high_cat)
    
    hybrid = low_dog + high_cat 
    cv.imshow('Hybrid Image', hybrid)
    
    ryan = cv.imread(r'ryan.jpg')
    ryan = img_as_float(ryan)
    clara = cv.imread(r'clara.jpg')
    clara = img_as_float(clara)
    
    low_ryan = cv.filter2D(ryan, -1, low_pass)
    high_clara = cv.filter2D(clara, -1, high_pass)
    bonus = low_ryan + high_clara 
    cv.imshow('Bonus Image', bonus)

    cv.waitKey(0)

    
if __name__=="__main__":
    main()
    