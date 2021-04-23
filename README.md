# Traffic-Signs-Detection-Tensorflow-YOLOv3-YOLOv4
Final Project for Applications of Machine Learning in Mechantronic Systems


## About ##
There are two primary tasks for any recognition system, detection (finding the location and size of the object on the input image), and classification (classifying the detected objects into subclasses). Both tasks are usually done with a single detection/classification model such as YOLO or SSD, where input images are labelled with the bounding boxes and respective classes. However, labelling and training such datasets requires a lot of time and effort. Therefore the main goal of this project is to detect just one main class (signs), and integrate a custom built Convolutional Neural Network to classify detected objects (subclasses such as speed limit, stop signs etc). This way, we just need to train a detection model once, to detect one main class, while multiple classification models can be trained to classify detected objects as per task requirements.

![alt text](https://github.com/Alzaib/Traffic-Signs-Detection-Tensorflow-YOLOv3-YOLOv4/blob/main/images/flow.png)

## Output  ##
### Sign Detection and Classification ###
![alt text](https://github.com/Alzaib/Traffic-Signs-Detection-Tensorflow-YOLOv3-YOLOv4/blob/main/images/output_images/1.jpg)
![alt text](https://github.com/Alzaib/Traffic-Signs-Detection-Tensorflow-YOLOv3-YOLOv4/blob/main/images/output_images/2.jpg)
![alt text](https://github.com/Alzaib/Traffic-Signs-Detection-Tensorflow-YOLOv3-YOLOv4/blob/main/images/output_images/6.jpg)


## Setup and Requirement ## 
OpenCV (versoin 4)

Python 

Numpy

TensorFlow 

Keras

## YOLO Implementation ##

For this project, due to time constraints, we decided to use a publicly available dataset (german traffic signs) to train YOLO on our custom dataset which can be found here [1].  The signs in this dataset are divided into 4 main classes (prohibitory, danger, mandatory and other). As relabeling the images into 1 class is a time consuming process, so we decided to use the dataset with 4 classes. The custom YOLOv3 and YOLOv4-Tiny are trained on Google Colab. 

## Classification Model ##

This part involves building a custom convolutional neural network to classify between 43 classes of traffic signs. The dataset used for this part can be found here [2]. Additional methods to balance and expand the dataset are used for better predicted output. This model is also trained on Google Colab 

## Implementation on Jetson Nano ##

## Weight ##
Weight: https://drive.google.com/drive/folders/1cJl0CUJXfGHbd7LQWa1pcOIzKLrf2jdf?usp=sharing

Dataset (YOLO Format): https://www.kaggle.com/valentynsichkar/traffic-signs-dataset-in-yolo-format

Dataset (Classification): https://drive.google.com/drive/folders/15ZFDX9nNrkwMIas9FUO0V1AaiXquztCl?usp=sharing 

## References ## 

[1] Dataset (YOLO Format): https://www.kaggle.com/valentynsichkar/traffic-signs-dataset-in-yolo-format

[2] Dataset (Classification): https://drive.google.com/drive/folders/15ZFDX9nNrkwMIas9FUO0V1AaiXquztCl?usp=sharing 

[3] YOLO GoogleColab Tutorial: https://www.youtube.com/watch?v=_FNfRtXEbr4&t=703s 
