#!/usr/bin/env python

import numpy as np
import cv2
import cv2.cv as cv
from video import create_capture
import math

def Gaussian1D(GD, mean_a, mean_b, var, num):
    temp[256]

    for i in range(0,num):
        if i < mean_a:
            temp[i] = math.exp(-1*(i-mean_a) * (i-mean_a)/var)
        elif (i >= mean_a) & (i <=mean_b):
            temp[i] = 1.0
        else:
            temp[i] = math.exp(-1*(i-mean_a) * (i-mean_a)/var)
    
    min = 1.0
    max = 0.0

    for i in range(0, num):
        if temp[i] < min:
            min = temp[i]
        if temp[i] > max:
            max = temp[i]

    mag = max - min

    for i in range(0, num):
        GD[i]=(((temp[i]-min)/mag)*255);
 
def nothing(*arg):
    pass

def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)

       
def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
   


cv2.namedWindow('facedetect')
cv2.createTrackbar('off/on/start','facedetect',0,2,nothing)

if __name__ == '__main__':
    
    import sys, getopt
    
    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try: video_src = video_src[0]
    except: video_src = 0
    args = dict(args)
    
    cascade_fn = args.get('--cascade', "../data/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "../data/haarcascade_eye.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)
    
    
    cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')
    b = 0
    while True :

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = detect(gray, cascade)
        vis = img.copy()

        b = cv2.getTrackbarPos('off/on/start', 'facedetect')

        cnt=0

        if (b == 1):
            draw_rects(vis, rects, (0, 255, 0))
            #print(rects)
            #print(rects[0][0],rects[0][1],rects[0][2],rects[0][3],"asdf")
            
        if (b == 2):
            #draw_rects(vis, rects, (0, 255, 0))
            if(len(rects) != 0):
                
                rgb=100000000
                for i in range(rects[0][1], rects[0][3]):
                    for j in range(rects[0][0], rects[0][2]):

                        red=vis[i,j,2]
                        green=vis[i,j,1]
                        blue=vis[i,j,0]
                            
                        rgb=red+green+blue
                        
                        if rgb <= 10:
                            vis[i,j] = 0
                        else:
                            if(red > 10):
                                red=red*256/rgb
                                vis[i,j,2]=red
                            else:
                               vis[i,j,2]=0
                            if(green > 10):
                                green=green*256/rgb
                                vis[i,j,1]=green
                            else:
                                vis[i,j,1]=0
                            if(blue > 10):
                                blue=blue*256/rgb
                                vis[i,j,0]=blue
                            else:
                               vis[i,j,0]=0
                            #vis[i][j] = red * green / 255 ;

                        #print()
                        #if(blue >100):
                            #vis[i,j]=0
                        if(green > 100) & (red > 50) & (blue > 150):
                            vis[i,j]=255
                        else:
                            vis[i,j]=0
                       
                            
   
                        '''
                        if():
                            vis[i,j]=0
                            '''
                        #print(red+blue+green)
                       
                #print("asdfasdfasdf")
        
        
        cv2.imshow('facedetect', vis)
        
        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
