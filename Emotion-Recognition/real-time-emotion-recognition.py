#demo for real-time emotion recognition 
from deepface import DeepFace
import cv2

#opening webcam/making sure webcam is opened
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

#using OpenCV's pre-trained classifier for face and eyes
face_trained = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_trained = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    success, img = cap.read()
    #use pre-built facial attribute analysis to output age, gender, emotion, and race
    demography = DeepFace.analyze(img, enforce_detection=False)

    print("Age: ", demography["age"])
    print("Gender: ", demography["gender"])
    print("Emotion: ", demography["dominant_emotion"])
    print("Race: ", demography["dominant_race"])

    #convert image/video to greyscale
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #use detectMultiScale function to detect faces and store in a variable which holds the coordinates
    face_coordinates = face_trained.detectMultiScale(greyscale)

    #loop through the face coordinates and put boxes around faces and eyes
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),2)
        #display dominant emotion within the face box
        cv2.putText(img, demography["dominant_emotion"], (x+10,y+h-10), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        eyes = eye_trained.detectMultiScale(greyscale)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    #display webcam in a separate window
    cv2.imshow('webcam', img)
    
    #waits until q is pressed to end program
    c = cv2.waitKey(1) & 0xFF
    if c == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

