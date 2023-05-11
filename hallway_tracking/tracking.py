import cv2
import numpy as np
import torch
from sort import *

# read homography matrix
with open('001.txt', 'r') as f:
    homog_mat = np.array([float(n) for n in f.readline().split(',')]).reshape(3,3)

# load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

N = 100  # length of path

video_path = '001.avi'  # path to video

cap = cv2.VideoCapture(video_path)  # create video capture object
 
if (cap.isOpened()== False):  # ensure that file was opened
    print("Error opening video file!!!")

tracker = Sort(max_age=5)  # init tracker

ret = 0  # init flag and frame array
frame = 0

f = 5  # number of frames to skip

trace = {}  # coordinates of tracked objects
colors = {}  # colors for visualization

while(cap.isOpened()):
    for _ in range(f):
        ret, frame = cap.read()  # read every f-th frame

    if ret == True:
        preds = model(frame, size=640).xyxy[0].numpy()  # get predictions
        detections = tracker.update(preds[:, :5])  # track detections
        up_view = (np.ones(frame.shape) * 255.0).astype(np.uint8)  # create base for up view
        for d in detections:
            if d[4] not in trace:  # add trace array and random color if ID is new
                trace[d[4]] = []
            if d[4] not in colors:
                colors[d[4]] = tuple(np.random.random(size=3) * 256)
            if len(trace[d[4]]) > N:
                trace[d[4]].pop(0)
            trace[d[4]].append([int((d[0] + d[2]) / 2), int(d[3])])  # add point to trace
            for p in trace[d[4]]:
                frame = cv2.circle(frame, p, 
                               1, colors[d[4]], -1)  # draw path
                # get coordinates on ground plane
                gp = np.matmul(homog_mat, np.array(p + [1,]).reshape(-1, 1)).flatten()
                gp[0] /= -gp[2]
                gp[1] /= gp[2]
                gp = (gp*(2) + 300).astype(int)  # scale coordinates
                up_view = cv2.circle(up_view, (gp[0], gp[1]), 
                               1, colors[d[4]], -1)  # draw path on ground plane
            frame = cv2.putText(frame, 'ID %i' % d[4], 
                                (int((d[0] + d[2]) / 2), int(d[3]) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[d[4]],
                                2, cv2.LINE_AA)  # put text with ID
        frame = np.concatenate((frame, up_view), axis = 1)  # concat frame and ground plane
        cv2.imshow('frame', frame)  # show frame
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break  # break if q pressed
    else: 
        break
    
    
    
# clean up
cv2.destroyAllWindows()
cap.release()
