#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 06:43:24 2022

@author: rosemary
"""

import cv2 as cv
import numpy as np


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

width_number_of_squares = 6
height_number_of_squares = 8

objp = np.zeros((height_number_of_squares*width_number_of_squares,3), np.float32)
objp[:,:2] = np.mgrid[0:width_number_of_squares,0:height_number_of_squares].T.reshape(-1,2)

objpoints = []
imgpoints = []

cap = cv.VideoCapture(2)

while(cap.isOpened()):
    
    ret, frame = cap.read()
    
    #Ubah citra BGR ke Grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
    
    #Mencari sudut papan catur
    ret2, corners = cv.findChessboardCorners(gray, (width_number_of_squares,height_number_of_squares), None)
    
    
    if ret2 == True:
      objpoints.append(objp)
            
      corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
      imgpoints.append(corners2)
            
        
      img = cv.drawChessboardCorners(frame, (width_number_of_squares,height_number_of_squares), corners2, ret2)
            
        
    
    cv.imshow("Kalibrasi Kamera", frame)
    
    # Tahan frame selama 1ms
    if cv.waitKey(1) & 0xFF == ord('q'): 
      cv.destroyAllWindows()
      break
        
        
# Lepas capture dan tutup semua jendela
cap.release()
cv.destroyAllWindows()


#Lakukan Kalibrasi 
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(ret, mtx, dist, rvecs, tvecs)
print('Kalibarsi Selesai')

list_data=[]

list_data.append([ret, mtx, dist, rvecs, tvecs])

data = str(list_data)
with open('/home/rosemary/Documents/Python Open CV/hasil_kalibrasi.txt', 'w') as f:
   f.write(data)
