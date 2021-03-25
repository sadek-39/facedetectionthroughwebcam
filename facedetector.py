import cv2
from random import randrange

train_face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
#print(train_face);
#img=cv2.imread('test2.jpg')
webcam=cv2.VideoCapture(0)

while True:
    successful_frame_read,frame=webcam.read()
    #convert balck and white
    
    grayscale_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);
    face_coordinates=train_face.detectMultiScale(grayscale_img)
    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),10)
    cv2.imshow('Gray',frame);
    key=cv2.waitKey(1);
    if key==81 or key==113:
        break;

webcam.release()        
    
#image show
#cv2.imshow("this is sadek",img)
#cv2.waitKey()



#face_coordinates=train_face.detectMultiScale(grayscale_img)
#for(x,y,w,h) in face_coordinates:
#    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),10)


#print(x);
#print(face_coordinates)

