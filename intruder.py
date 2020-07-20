import numpy as np
import argparse
import sys
import cv2
# import math as m
from math import sqrt,pow

def get_intersection(a, b):
    # Intersection box coordinates
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    # Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # when no overlap happens
    if (width<0) or (height <0):
        return 0.0
    area_overlapped = width * height
    # Combined area
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlapped
    # Overlapped area ratio
    ratio = area_overlapped / (area_combined + (1e-5)  )
    return ratio

mouse_pts = []
def get_mouse_points(event, x, y, flags, param):
    global pointX, pointY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        pointX, pointY = x, y
        cv2.circle(image, (x, y), 5, (255, 255, 0), 10)
        mouse_pts.append((x, y))

# Parse the arguments from command line
arg = argparse.ArgumentParser(description='Social distance detection')
arg.add_argument('-v', '--video', type = str, default = 'C:/Users/Admin/Desktop/intruder_in_roi/3_view.mp4', help = 'Video file path. If no path is given, video is captured using device.')
arg.add_argument('-m', '--model', required = True, help = "Path to the pretrained model.")
arg.add_argument('-p', '--prototxt', required = True, help = 'Prototxt of the model.')
arg.add_argument('-l', '--labels', required = True, help = 'Labels of the dataset.')
arg.add_argument('-c', '--confidence', type = float, default = 0.2, help='Set confidence for detecting objects')
args = vars(arg.parse_args())

labels = [line.strip() for line in open(args['labels'])]

# Load model
print("\nLoading model...\n")
network = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
print("\nStreaming video using device...\n")

writer=None

# Capture video from file or through device
if args['video']:
    cap = cv2.VideoCapture(args['video'])
else:
    cap = cv2.VideoCapture(0)

cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)
frame_no = 0
first_frame_display = True

while cap.isOpened():
    frame_no = frame_no+1
    # Capture one frame after another
    ret, frame = cap.read()

    if not ret:
        break

    if frame_no == 1:
        while True:
            image = frame
            cv2.imshow("image", image)
            cv2.waitKey(1)
            if len(mouse_pts) == 4:
                cv2.destroyWindow("image")
                break
            first_frame_display = False
        print("ROI points", mouse_pts)


    pts = np.array(
    [mouse_pts[0], mouse_pts[1], mouse_pts[3], mouse_pts[2]], np.int32
    )

    cv2.polylines(frame, [pts], True, (255, 255, 0), thickness=4)

    (h, w) = frame.shape[:2]

    # Resize the frame to suite the model requirements. Resize the frame to 300X300 pixels
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    network.setInput(blob)
    detections = network.forward()

    coordinates = dict()

    # Focal length
    F = 615

    warped_pts = []

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:

            class_id = int(detections[0, 0, i, 1])

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # Filtering only persons detected in the frame. Class Id of 'person' is 15
            if class_id == 15.00:

                coordinates[i] = (startX, startY, endX, endY)

                threshold_value=0.07
            
                roi=(mouse_pts[1][0], mouse_pts[1][1], mouse_pts[2][0], mouse_pts[2][1])
                rect_pts=(startX,startY,endX,endY)

                ratio = get_intersection(roi,rect_pts)

                if (ratio > threshold_value):
                    print("Intruder detected")
                    cv2.rectangle(frame, (startX,startY), (endX, endY),(0, 0,255), 2)

    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter('output.mp4', fourcc, 30,(frame.shape[1], frame.shape[0]), True)

    # Show frame
    cv2.imshow('Frame', frame)
    cv2.resizeWindow('Frame',800,600)

    writer.write(frame)

    key = cv2.waitKey(1) & 0xFF

    # Press `q` to exit
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
