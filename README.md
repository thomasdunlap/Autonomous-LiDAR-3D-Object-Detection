
# Autonomous Perception: Tracking 3D-Objects Over Time


![Gif of 60 frames of darknet](movie/darknet.gif)

Camera-lidar sensor fusion detection takes four steps:
  1. Computing LiDAR point-clouds from range images.
  2. Transforming the point-cloud to a Bird's Eye View using the Point Cloud Library (PCL).
  3. Using both YOLO3 Darknet and Resnet to predict 3D dectections on the combined camera and lidar images.
  4. Evaluating the detections based Precision and Recall.  


## 1. Computing Lidar Point-Clouds from Waymo Range Images

Waymo uses Lidar, cameras, Radar for autonomous navigation.  They even microphones to help detect ambulances and police cars.

![](img/waymo_lidar.png)


Roof-mounted "Top" lidar rotates 360 degrees on top of roof with a vertical field of vision -17.6 degrees to +2.4 degrees with a 75m limit in the dataset. Also space between lidar beams widens with distance.  Limitations of 360 lidar include the space between beams (aka resolution) widening with distance from the origin.  Also the car chasis will create blind spots, creating the need for Perimeter LiDAR sensors:

![](img/top_lidar_blind_spot.png)

Perimeter LiDAR has vertical field of vision from -90 degrees to + 30 degrees at up to 20 meters.  Actual sensor range is higher, but the dataset limits at 20m.  Perimeter lidars are on front, back and front corners of vehicle.

There are 64 LEDs in Waymo's top LiDAR sensor.  Extrinsic Calibration Matrix top lidar +1.43m from origin of vehicle coordinate system with height fo +2.184.

### Visualizing LiDAR Range and Intensity Channels

![Lidar visualization 1](img/range_img0.png)

Lidar data is stored as a range image in the Waymo Open Dataset. Using OpenCV and NumPy, we filtered the "range" and "intensity" channels from the image, and converted the float data to 8-bit unsigned integers.  Below is a visualization of two video frames, where the top half is the range channel, and the bottom half is the intensity for each visualization: 

![Lidar visualization 2](img/range_img1.png)

### Visualizing th LiDAR Point-cloud

![](img/ScreenCapture_2021-12-14-21-16-15.png)

We leveraged the Open3D library to make a 3D interactive visualization of the lidar point-cloud.  Commonly visible features are windshields, tires, and mirros within 40m. Beyond 40m, cars are like slightly rounded rectangles where you might be able to make ou the windshield.  Further away vehicles and extremely close vehicles typically have lower resolution, as well as vehicles obstructing the detection of other vehicles.

**10 Vehicles Showing Different Types of Lidar Interaction:**
![](img/midterm-vehicles-cloud.png)

1. Truck with trailer - most of truck is high resolution visible, but part of the trailer is in the 360 lidar's blind-spot.
2. Car partial in blind spot, back-half isn't picked up well.  This car blocks the larges area behind it from being detected by the lidar.
3. Car shape is higly visible, where you can even see the side-mirrors and the lidar passing through the windshield.
4. Car driving in other lane.  You can see the resolution of the car being lower because the further away the 64 LEDs project the lasers, the futher apart the points of the cloud will be.  It is also obstructed from some lasers by Car 2.
5. This parked is unobstructed, but far enough away where it's difficult to make our the mirrors or the tires.
6. Comparing this car to Car 3, you can see where most of the definition is either there or slightly worse, because it is further way.
7. Car 7 is both far away and obstructed, so you can barely tell it's a car.  It's basically a box with probably a windshield.
8. Car 8 is similar to Car 6 on the right side, but obstructed by Car 6 on the left side. 
9. Car 9 is at the limit of the lidar's dataset's perception.  It's hard to tell it's a car.
10. Car 10 is at the limit of the lidar's perception, and is also obstructed by car 8.

[comment]:![](img/ScreenCapture_2021-12-14-21-26-37.png)

[comment]:![](img/ScreenCapture_2021-12-14-21-32-00.png)

[comment]:![](img/ScreenCapture_2021-12-14-21-33-30.png)

[comment]:![](img/ScreenCapture_2021-12-14-21-35-05.png)

## 2.  Transforming the point-cloud to a Bird's Eye View using the Point Cloud Library (PCL)

### Convert sensor coordinates to Bird's-Eye View map coordinates 

![](img/s2e1a.png)

The birds-eye view (BEV) of a LiDAR point-cloud is based on the transformation of the x and y coordinates of the points.

BEV map properties:
* Height:

  ![H_{i,j} = max(P_{i,j} \cdot [0,0,1]T)](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+H_%7Bi%2Cj%7D+%3D+max%28P_%7Bi%2Cj%7D+%5Ccdot+%5B0%2C0%2C1%5DT%29)

* Intensity: 

  ![I_{i,j} = max(I(P_{i,j}))](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+I_%7Bi%2Cj%7D+%3D+max%28I%28P_%7Bi%2Cj%7D%29%29)

* Density:

  ![D_{i,j} = min(1.0,\ \frac{log(N+1)}{64})](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+D_%7Bi%2Cj%7D+%3D+min%281.0%2C%5C+%5Cfrac%7Blog%28N%2B1%29%7D%7B64%7D%29)

![P_{i,j}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+P_%7Bi%2Cj%7D) is the set of points that falls into each cell, with ![i,j](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+i%2Cj) as the respective cell coordinates. ![N_{i,j}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+N_%7Bi%2Cj%7D) refers to the number of points in a cell.

![](img/s2e1b.png)

### Compute intensity layer of the BEV map

![](img/s2e2a.png)


We created a BEV map of the "intensity" channel from the point-cloud data. We identified the top-most (max height) point with the same (x,y)-coordinates from the point-cloud, and assign the intensity value to the corresponding BEV map point.  The data was normalized and outliers were removed until the features of interest were clearly visible. 

![](img/s2e2b.png)

### Compute height layer of the BEV map 

![](img/s2e3a.png)

This is a visualization of the "height" channel BEV map.  We sorted and pruned point-cloud data, normalizing the height in each BEV map pixel by the difference between max. and min.

![](img/s2e3b.png)

## 3. Model-based Object Detection in BEV Image

We used YOLO3 and Resnet deep-learning models to doe 3D Object Detection.  Complex-YOLO: Real-time 3D Object Detection on Point Clouds and Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds.

### Extract 3D bounding boxes from model response 


The models take a three-channel BEV map as an input, and predict the class about coordinates of objects (vehicles).  We then transformed these BEV coordinates back to the vehicle coordinate-space to draw the bounding boxes in both images.


Below is a gif the of detections in action:
![Results from 50 frames of resnet detection](movie/detection2.gif)


## 4. Performance Evaluation for Object Detection

### Compute intersection-over-union between labels and detections 

The goal of this task is to find pairings between ground-truth labels and detections, so that we can determine wether an object has been (a) missed (false negative), (b) successfully detected (true positive) or (c) has been falsely reported (false positive). Based on the labels within the Waymo Open Dataset, your task is to compute the geometrical overlap between the bounding boxes of labels and detected objects and determine the percentage of this overlap in relation to the area of the bounding boxes. A default method in the literature to arrive at this value is called intersection over union, which is what you will need to implement in this task.

After detections are made, we need a set of metrics to measure our progress. Vehicle predictions vs ground-truth labels typically fall into 4 categories:

* *TP*: True Positive - Predicts vehicle or other object is there correctly
* *TN*: True Negative - Correctly predicts vehicle or object is not present
* *FP*: False Positive - Dectects object class incorrectly
* *FN*: False Negative - Didn't detect object class when there should be a dectection


### Classification Metrics Based on Precision and Recall 

After all the Lidar and Camera data has been transformed, and the detections have been predicted, we calculate the following metrics for the bounding box predictions:

#### Formulas

* *Precision*:

    ![\frac{TP}{TP + FP}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cfrac%7BTP%7D%7BTP+%2B+FP%7D)


* *Recall*: 

    ![\frac{TP}{TP + FN}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cfrac%7BTP%7D%7BTP+%2B+FN%7D)

* *Accuracy*: 
    
    ![\frac{TP + TN}{TP + TN + FP + FN}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cfrac%7BTP+%2B+TN%7D%7BTP+%2B+TN+%2B+FP+%2B+FN%7D)


* *Mean Average Precision*: 
    
    ![\frac{1}{n} \sum_{Recall_{i}}Precision(Recall_{i})](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cfrac%7B1%7D%7Bn%7D+%5Csum_%7BRecall_%7Bi%7D%7DPrecision%28Recall_%7Bi%7D%29)

### mAP with Precision and Recall Visualizations

Results from 50 frames:
![Results from 50 frames](img/metricsMult.png)


