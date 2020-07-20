
# Intruder Detection-using-OpenCV

## Set Up

To begin with, use the requirements.txt file to install the dependencies.
```
pip install -r requirements.txt
```

## Saving Videos

## Modify the video location in the default argument:

For example:
```
arg.add_argument('-v', '--video', type = str, default = 'C:/Users/Admin/Desktop/intruder_in_roi/3_view.mp4', help = 'Video file path. If no path is given, video is captured using device.') 
```

## Execute

Use the following command to execute the code.
```
python intruder.py --prototxt SSD_MobileNet_prototxt.txt --model SSD_MobileNet.caffemodel --labels class_labels.txt
```

Running the above command will open a window of the first frame in the video. At this point the code expects the user to mark 6 points by clicking appropriate positions on the frame.

#### 4 points:
The 4 points are used to mark the Region of Interest (ROI) where you want to monitor.
These 4 points need to be provided in a pre-defined order which is following.

* __Point1 (bl)__: Bottom left
* __Point2 (br)__: Bottom right
* __Point3 (tl)__: Top left
* __Point4 (tr)__: Top right


### To do:
Saving Video using ```write```
