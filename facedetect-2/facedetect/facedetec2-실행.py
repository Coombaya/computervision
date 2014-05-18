#!/usr/bin/env python

import numpy as np
import cv2
import cv2.cv as cv
from video import create_capture


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
        red=0
        blue=0
        green=0
        cnt=0

        if (b == 1):
            draw_rects(vis, rects, (0, 255, 0))
            #print(rects)
            #print(rects[0][0],rects[0][1],rects[0][2],rects[0][3],"asdf")
            
        if (b == 2):
            draw_rects(vis, rects, (0, 255, 0))
            if(len(rects) != 0):
                for x1, y1, x2, y2 in rects:
                    for i in range(y1, y2):
                        for j in range(x1, x2):
                            red=vis[i,j,2]
                            green=vis[i,j,1]
                            blue=vis[i,j,0]
                            #print(vis[i,j])
                            if(red > 20):
                               vis[i,j,2]=red*255/(red+green+blue)
                            else:
                               vis[i,j,2]=0
                            if(green > 20):
                               vis[i,j,1]=green*255/(red+green+blue)
                            else:
                               vis[i,j,1]=0
                            if(blue > 20):
                               vis[i,j,0]=blue*255/(red+green+blue)
                            else:
                               vis[i,j,0]=0

        cv2.imshow('facedetect', vis)
        
        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
