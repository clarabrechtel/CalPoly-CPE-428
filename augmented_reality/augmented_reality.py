# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:50:03 2020

@author: Clara Brechtel 
"""

import cv2 as cv 
import numpy as np
from skimage.util import img_as_float

def imageSIFT(img):
    gry_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gry_img, None)
    img = cv.drawKeypoints(gry_img, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img, kp, des

def BFMatch(query, query_kp, query_des, train, train_kp, train_des):
    
    bf = cv.BFMatcher(cv.NORM_L2)
    matches = bf.knnMatch(query_des, train_des, k=2)
    keep = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            keep.append(m)
            
    matched = cv.drawMatches(query, query_kp, train, train_kp, keep, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
    
    return matched, keep
    
def fitHomography(query, query_kp, train, train_kp, keep):
    
    query_pts = np.float32([query_kp[m.queryIdx].pt for m in keep]).reshape(-1,1,2)
    train_pts = np.float32([train_kp[m.trainIdx].pt for m in keep]).reshape(-1,1,2)
    
    H, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 10.0)
    inliers = 0 
    for val in mask:
        inliers += sum(val) 
    perc_inlier = inliers/len(mask)
    
    matchesMask = mask.ravel().tolist()
    
    w,h,z = query.shape
    pts = np.float32([ [0,0], [0,w-1], [h-1,w-1], [h-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts, H)
    
    train = cv.polylines(train, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    
    matched_img = cv.drawMatches(query, query_kp, train, train_kp, keep, None, 
                                 matchColor = (0,255,0), matchesMask = matchesMask, 
                                 flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    return matched_img, perc_inlier, H

def overlayImage(overlay, frame, H):
    
    frame = img_as_float(frame)
    overlay = img_as_float(overlay)
    x, y, z = overlay.shape
    w, h, z = frame.shape 
    
    mask = np.ones(shape=[x, y, 3], dtype=np.float64)
    warped = cv.warpPerspective(overlay, H, (h, w))
    alpha = cv.warpPerspective(mask, H, (h, w))
    
    overlay_mask = np.multiply(warped, alpha)
    inverse_warped = 1 - alpha 
    frame_mask = np.multiply(frame, inverse_warped) 
    result = np.add(overlay_mask, frame_mask)
    
    return result 

if __name__=="__main__":
    stones = cv.imread(r'stones.png')
    overlay = cv.imread(r'overlay.png')
    stones_sift = stones.copy()
    stones_match = stones.copy()
    stones_sift, stones_kp, stones_des = imageSIFT(stones_sift)
    cv.imshow('Stones with Keypoints', stones_sift)
    cv.waitKey(0)
    
    average = []
    vid = cv.VideoCapture('input.mov')
    while vid.isOpened():
        read, frame = vid.read()
        if not read:            # Breaks out of loop if frame can't be read
            break
        frame_sift = frame.copy()
        frame_match = frame.copy()
        frame_sift, frame_kp, frame_des = imageSIFT(frame_sift)
        matched_frame, keep = BFMatch(stones_sift, stones_kp, stones_des, frame_sift, frame_kp, frame_des)
        masked_matches, perc_inlier, H = fitHomography(stones_match, stones_kp, frame_match, frame_kp, keep)
        average.append(perc_inlier)
        augmented_frame = overlayImage(overlay, frame, H)
        
        cv.imshow('Video with SIFT Features', frame_sift)
        cv.waitKey(1)
        cv.imshow('Video Frame and Target Image Matched', matched_frame)
        cv.waitKey(1)
        cv.imshow('Matched Features with Mask', masked_matches)
        cv.waitKey(1)
        cv.imshow('Video Augmented with Overlay', augmented_frame)
        cv.waitKey(1)
        
    vid.release()
    cv.waitKey(0)
    average = sum(average)/len(average)
    print('Average percentage of inliers per frame: {}'.format(average))  
    
    
    
    