# library
import cv2
import sys
from keras.models import load_model

# model 
model = load_model('model.h5')
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

expression = {0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Sad",5:"Surprise",6:"Neutral"}

cap = cv2.VideoCapture(0)


while(True):
    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(1,1),flags = cv2.CASCADE_SCALE_IMAGE)

    print("Detected Faces",len(faces))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,0), 2)
        saveImage = gray[y:y+h,x:x+w]
        saveImage = cv2.resize(saveImage,(48,48))
        res = model.predict_classes(saveImage.reshape(1,48,48,1))[0]
        print(res)
        cv2.putText(img,expression.get(res),(x,y),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=2,color=(0,255,255),thickness=2)
    cv2.imshow("Expression Detected", img)
    if cv2.waitKey(1) == 27:
        break
