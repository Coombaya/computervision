#!/usr/bin/env python

import numpy as np
import cv2
import cv2.cv as cv
from video import create_capture
import wx

def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)

#얼굴찾는함수          
def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects
#이미지에 사각형을 그리는 함수
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
     
if __name__ == '__main__':
    
    import sys, getopt  
    #윈도우이름
    cv2.namedWindow('facedetect')
    
    
    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try: video_src = video_src[0]
    except: video_src = 0
    args = dict(args)
    #얼굴과 눈 xml파일 입력
    cascade_fn = args.get('--cascade', "../data/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "../data/haarcascade_eye.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)
    
    #카메라 이미지받아온다
    cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')
    #onoff = 1
    
    while True :
        
        #onoff = cv2.getTrackbarPos('switch','facedetect')
        ret, img = cam.read()#카메라이미지를 받는다
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
       
        
        rects = detect(gray, cascade)#얼굴을 찾은 범위를 저장
        vis = img.copy()
        #if onoff == 1 :
        draw_rects(vis, rects, (0, 255, 0))#사각형으로 얼굴을 표시
            
        cv2.imshow('facedetect', vis)#얼굴을 찾은 사각형을 표시     
        
        if 0xFF & cv2.waitKey(5) == 27:#esc누르면 종료
            break
    cv2.destroyAllWindows()
