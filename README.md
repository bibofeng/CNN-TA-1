# Image procesing in Finance

## Abstract

Computational intelligence techniques for financial trading systems have always been quite popular.
In the last decade, deep learning models start getting more attention, especially within the image
processing community.

In this study, we propose a novel algorithmic trading model CNN-TA using a 2-D Convolutional
Neural Network based on image processing properties. In order to convert financial time series into
2-D images, 15 different technical indicators each with different parameter selections are utilized.
Each indicator instance generates data for a 15 day period. As a result, 15x15 sized 2-D images are
constructed. Each image is then labelled as Buy, Sell or Hold depending on the hills and valleys of
the original time series.

The results indicate that when compared with the Buy Hold Strategy and other common trading
systems over a long out-of-sample period, the trained model provides better results for stocks and
ETFs.


## Introduction

### 1.1 Overview

In this study, we proposed a novel approach that converts 1-D financial time series into a 2-D image-
like data representation in order to be able to utilize the power of deep convolutional neural network 
for an algorithmic trading system. In recent years, deep learning based prediction/classification
models started emerging as the best performance achievers in various applications, outperforming
classical computational intelligence. However, image processing and vision based problems dominate
the type of applications that these deep learning models outperform the other techniques.
Nowadays, deep learning methods have started appearing on financial studies. 

There are some implementations of deep learning techniques such as Recurrent neural network (RNN), convolutional 
neural network (CNN), and long short term memory (LSTM).CNNs have been by far, the 
most commonly adapted deep learning model. Meanwhile, majority of the CNN implementations in
the literature were chosen for addressing computer vision and image analysis challenges.

### 1.2 Problem Statement
Take the data of any company And then use machine learning models to predict the future
returns using present time-series data. Basically the idea is we convert the times series data into a
2D image and then process it in our CNN model and analyse the profit.


## Method - Image Processing
### 2.1 Data Extraction
In our study, the daily stock prices of various firms or company are obtained from finance.yahoo.com
for training,validation and testing purposes.

### 2.2 Labelling
Once the extraction of data is done , we move on to the labelling part where each stock is manually
marked as Buy,Sell or Hold depending on the top and bottom point in the sliding window approach.
In this approach the bottom point is marked as Buy since it is the least price encountered in the
sliding window,the top point is marked as Sell to maximise the profit, whereas the rest are marked as
Hold.Once the labelling is done we move on to the image creation part.

<!-- Labling Method Image -->

### 2.3 Image Creation
For each day a (15×15) image is generated by using 15 technical indicators and 15 different intervals
of technical indicators. Meanwhile, each image uses the associated label (”HOLD” , ”BUY” , ”SELL”)
with the sliding window logic. 

The order of the indicators is important, since different orderings will result in different image formations.
To provide a consistent and meaningful image representation, we clustered indicator groups (oscillator or trend) and similar behaving
indicators together or in close proximity and normalized all the indicators.

#### 2.3.1 Technical Indicator
In image creation phase, for each day, RSI, Williams %R, WMA, EMA, ,Triple EMA, CCI, CMO,
MACD, PPO, ROC, and PSI values for different intervals (6 to 20 days) are calculated using TA-lib
library.
Since 6 to 20 days of indicator ranges are used in our study, swing trades for 1 week to 1
month periods are focused. Different indicator choices and longer ranges can be chosen for models
aiming for less trades.

#### 2.3.2 Normalization
Since the values of each indicator varies significantly from each other.We normalized the value of all
indicators so that there was no big difference in pixels of images.As the value of all indicators are normalized 
between 0 to 1. Now we are in stage to create the images using PIL Library.

<!-- Normalization Method Image -->

### 2.4 CNN
In the proposed algorithm the CNN model used for analysis phase consists of 8 layers namely:
- The Input Layer 
- Two Convolutional Layers 
- A Max Pooling Layer
- Two Dropout Layers
- A Fully connected Layer
- An Output Layer

<!-- Model Image -->

