import cv2
import numpy as np 

cap = cv2.VideoCapture(0)
f_cnt = 0
w_cnt = 0
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (256, 256))
    cv2.imshow('d', frame)
    key = cv2.waitKey(1)
    if key == ord('f'):
        print("taken")
        cv2.imwrite('./dataset/picture_{}.png'.format(f_cnt), frame)
        f_cnt = f_cnt + 1
    