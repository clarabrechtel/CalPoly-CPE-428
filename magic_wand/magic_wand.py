# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 11:56:09 2020

@author: clara
"""
import cv2 as cv 
import numpy as np
import argparse 

# Define intrinsic parameters of camera and ball 
pt_x = 134.875
pt_y = 239.875
focal = 485.82423388827533
radius = 3 
    
def main():
    # construct the argument parser and parse the arguments 
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
    	help="path to input video")
    args = vars(ap.parse_args())
    
    # load the input video
    vid = cv.VideoCapture(args["input"])
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        
        img_center, img_radius = detect_circles(frame) 
        X, Y, Z = compute_coordinates(img_center, img_radius, pt_x, pt_y, focal, radius)
        Z = int(Z)
        frame = cv.putText(frame, str(Z) + 'cm', img_center, cv.FONT_HERSHEY_SIMPLEX,.6, (0,255,0), 1, cv.LINE_AA)
        cv.imshow('Depth', frame)
        cv.waitKey(5)
        
        cube = get_cube(X, Y, Z, radius)
        a = []
        for coord in cube:
            proj = img_projection(pt_x, pt_y, focal, coord[0], coord[1], coord[2])
            a.append(proj)
        for i in range(0, len(a)-1):
            img = connect_points(frame, a[i], a[i+1])
        
        cv.imshow('Cube', img)
        cv.waitKey(5)
        
    vid.release()
    
def detect_circles(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray,(9,9),2)
    rows = blurred.shape[0]
    circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, 1, rows / 16,
                              param1=200, param2=30,
                              minRadius=1, maxRadius=100)
    circles = np.asarray(circles, dtype=np.uint16)
    for i in circles[0, :]:
        img_center = (i[0],i[1])
        img_radius = i[2]
        cv.circle(frame, img_center, img_radius, (0, 255, 0), 2)
        cv.imshow('Detected Circles', frame)
    cv.waitKey(5)        
    return img_center, img_radius 
    
    
def compute_coordinates(img_center, img_radius, pt_x, pt_y, focal, radius):
    camera_x = (img_center[0] - pt_x) / focal
    camera_y = (img_center[1] - pt_y) / focal
    z_depth = focal * (radius / img_radius)
    X = camera_x * z_depth 
    Y = camera_y * z_depth
    
    return X, Y, z_depth

def get_cube(X, Y, Z, R):
    cube1 = [X+R, Y+R, Z+R]
    cube2 = [X+R, Y-R, Z+R]
    cube3 = [X-R, Y-R, Z+R]
    cube4 = [X-R, Y+R, Z+R]
    cube5 = [X+R, Y+R, Z-R]
    cube6 = [X+R, Y-R, Z-R]
    cube7 = [X-R, Y-R, Z-R]
    cube8 = [X-R, Y+R, Z-R]
    cube = [cube1, cube2, cube3, cube4, cube1, cube5, cube6, cube7, cube8, cube5, cube6, cube2, cube3, cube7, cube8, cube4]

    return cube
    
# Function calculates the 2D projection of a 3D point 
def img_projection(pt_x, pt_y, focal, X, Y, Z):
    img_x = (focal * X / Z) + pt_x 
    img_y = (focal * Y / Z) + pt_y 
    proj = (int(img_x), int(img_y))
    
    return proj

def connect_points(frame, pt_1, pt_2):
    img = cv.line(frame, pt_1, pt_2, (0,255,0), 1, cv.LINE_8)
    
    return img 
    
if __name__=="__main__":
    main()
    
        
        