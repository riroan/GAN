import numpy as np
import cv2

cnt =0
num_images = 1938

def faceDetect(path):
    global cnt
    face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')
    
    try:
        cap = cv2.imread(path)
    except:
        print('카메라 로딩 실패')
        return
    frame = cap
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    
    for face in faces:
        (x,y,w,h) = face
        image = frame[y:y+h,x:x+w]
        image = cv2.resize(image,(128,128),cv2.INTER_LINEAR)
        cv2.imwrite('faces122/face'+str(cnt)+'.jpg',image)
        print('image saved ->'+'faces128/face'+str(cnt)+'.jpg')
        cnt+=1
    
for i in range(num_images):
    path = 'images/image'+str(i)+'.jpg'
    faceDetect(path)
