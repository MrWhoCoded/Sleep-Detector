import cv2
import numpy as np
import dlib
from math import hypot
import time
import playsound
import sys

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("face_landmarks.dat")

cap = cv2.VideoCapture(0)

blink = False
play = False
count = 0

def midpoint_determiner(p1, p2):
    return int((p1.x + p2.x)/2),int((p1.y + p2.y)/2)


font = cv2.FORMATTER_FMT_C

print("starting Detection...")
time.sleep(0.5)
while True:

    ret, frame = cap.read()

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(grey)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0,225,0), 3)

        landmarks = predictor(grey, face)

        # draws the different landmarks on the face
        """for i in range(0,68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            #cv2.circle(frame, (x, y), 3, (0,225,0), -1)"""

        eye1_left_mark = (landmarks.part(36).x, landmarks.part(36).y)
        eye1_right_mark = (landmarks.part(39).x, landmarks.part(39).y)
        eye2_left_mark = (landmarks.part(45).x, landmarks.part(45).y)
        eye2_right_mark = (landmarks.part(42).x, landmarks.part(42).y)

        cv2.line(frame, eye1_left_mark, eye1_right_mark, (0,225,0), 1)
        cv2.line(frame, eye2_left_mark, eye2_right_mark, (0,225,0), 1)

        top_center1 = midpoint_determiner(landmarks.part(37), landmarks.part(38))
        bottom_center1 = midpoint_determiner(landmarks.part(41), landmarks.part(40))
        top_center2 = midpoint_determiner(landmarks.part(43), landmarks.part(44))
        bottom_center2 = midpoint_determiner(landmarks.part(46), landmarks.part(47))

        cv2.line(frame, top_center1, bottom_center1, (0,225,0), 1)
        cv2.line(frame, top_center2, bottom_center2, (0,225,0), 1)

        hor_line_len1 = hypot((top_center1[0] - bottom_center1[0]), (top_center1[1] - bottom_center1[1]))
        vert_line_len1 = hypot((eye1_left_mark[0] - eye1_right_mark[0]), (eye2_left_mark[1] - eye2_right_mark[1]))

        ratio = int(vert_line_len1/hor_line_len1)
        
        # blink/sleep detection
        if ratio >= 4: #if the ratio gets less than 4, it implies the person is blinking/sleeping
            cv2.putText(frame, "BLINKING", (50,150), font, 5, (0,255,0))
            count += 1 #starts the timer(count)
            if count >=50:
                #print("Sleeping")
                playsound.playsound("foghorn-daniel_simon.mp3") #plays sound to wakeup the person
            else:
               continue
        
        #print(blink)
        blink = False
        #print(count)
        count = 0

    cv2.imshow("sleep detector", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break
