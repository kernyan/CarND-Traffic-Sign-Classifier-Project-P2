# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The main outline of this report are:
* Load Data and Visualization
* Preprocess and Augment Data
* Design and Test a Model Architecture
* Training the model
* Testing Final Model on New Images

[//]: # (Image References)

[image1]: ./examples/ShowImgLabel0.png "Speed_limit_20"
[image2]: ./examples/ShowImgLabel10.png "No_passing"
[image3]: ./examples/ShowImgLabel20.png "Dangerous_right_curve"
[image4]: ./examples/Histogram_X_train.png "Histogram_X_train"
[image5]: ./examples/Histogram_X_train_augmented.png "Histogram_X_train_augmented"
[image6]: ./examples/LeNet_20e_0.1b_0.001l.jpg "LeNet_HighL2Regularization"
[image7]: ./examples/LeNet_20e_0.01b_0.001l.jpg "LeNet_LowL2Regularization"
[image8]: ./examples/vgg6_40e_0.001b_0.0005l.jpg "VGG6_Loss"
[image9]: ./NewGerman/Processed_NewSigns.png "Processed Traffic Signs"
[image10]: ./examples/22BumpyRoad.png "22 Bumpy Road"

#### 1. Load Data and Visualization

```python
training_file   = './traffic-signs-data/train.p'
validation_file = './traffic-signs-data/valid.p'
testing_file    = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test  = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test   =  test['features'],  test['labels']
```

Number of training examples   = 34799
Number of validation examples = 4410
Number of testing examples    = 12630
Image data shape              = (32, 32, 3)
Number of classes             = 43

To ensure that the training images and the labels are indexed correctly, a function called `ShowImgAndLabelWithProcessed` is used to randomly show k images of a particular class. In addition, we also included two additional rows of images. The images below are samples of our visualization:

1. **Row1** Original image
2. **Row2** Histogram equalized
3. **Row3** Grayscaled and then Histogram equalized

![alt text][image1] ![alt text][image2] ![alt text][image3]

A histogram is also used to show the frequency of images we have in each class.

![alt text][image4]

#### Commentary

From the visualization above, we observed that Histogram Equalization `GeneralEqualizeHist` is very successful in correcting images that are too dark. On the other hand, by looking at the second and third rows of the images, it is hard to tell whether grayscale or keeping the RGB color channels are worth it. We eventually decided to keep RGB color channels after discovering that it provides superior fit over grayscale images on the [LeNet](#model-1-lenet) framework.

In addition, the histogram shows that the number of images varies widely across different labels. We decided to augment the data because we do not want our Neural Network's prediction to bias towards classes with higher number of images.

#### 2. Preprocess and Augment Data

We introduce random scaling `scaleImg` and random rotation `rotateImg`. For each class, if the number of observations is less than 1000, we randomly rotate the image by a degree between +/-20 degrees, and then randomly enlarge or shrink the image by 2 pixels (cropping or padding with zero to maintain the image shape at 32x32). See function `AugmentData` for implementation of data augmentation.

We considered random brightness and image flip as additional data augmentation techniques. However, we found out that the effect of uniform adjustment on brightness by a random amount on RGB is nullified after image normalization (see `NormalizeImg`). Whereas flipping the data upside down or left right does not make much sense, especially considering that 1) most signs are asymmetric (e.g., speed limit of 100 when flipped left-right becomes 001), and 2) after left-right flip, we will actually no longer be able to distinguish between label *19 - dangerous curve to the left* and label *20 - dangerous curve to the right*. As such, we excluded both random brightness and image flip in our data augmentation step.

Below we show a histogram of our augmented training data

![alt text][image5]

Although we could have generated more augmented data so that each classes have the same number of images, we decided that having at least a thousand images for each classes should be sufficient.

#### 3. Design and Test a Model Architecture

We tried several different networks, they are variants of both the LeNet and VGG networks. Here, we only show 3 of them as they are useful for the reader in understanding our development process.

#### Model 1 LeNet

This is a modified Convolutional Network from LeCun, Bottou, Bengio, and Haffner. "Gradient-Based Learning Applied to Document Recognition, Nov 1998". The main changes we made were the size of the final few fully connected layers. 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5   	| 1x1 stride, valid padding, outputs 10x10x16 	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Fully connected		| (5x5x16) 400x1024     						|
| RELU					|												|
| Dropout 50%			|												|
| Fully connected		| 1024x1024     								|
| RELU					|												|
| Dropout 50%			|												|
| Fully connected		| 1024x43			      						|
| Softmax				|           									|

#### Model 2 LeNet2

This is a model employing the 2-stage connection skipping idea from Sermanet, and LeCun. "Traffic Sign Recognition with Multi-Scale Convolutional Networks, July 2011". It is extending Model 1 by introducing a skip connection. 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5   	| 1x1 stride, valid padding, outputs 10x10x16 	|
| **(1)** Max pooling   | 2x2 stride,  outputs 5x5x16   				|
| Fully connected		| (5x5x16) 400x1024     						|
| RELU					|												|
| Dropout 50%			|												|
| Fully connected		| 1024x1024			     						|
| RELU					|												|
| **(2)** Dropout 50%	|												|
| Concat **(1), (2)**	| (5x5x16 => 400)	+ (1024) = 1424				|
| Fully connected		| 1424x43			      						|
| Softmax				|           									|

#### Model 3 VGG6

My final model is a modified VGG Convolutional Network from Simonyan, Zisserman (see function `VGG6`). It consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x128	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x128	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x128 	    			|
| Convolution 3x3	    | 1x1 stride, same padding, outputs  8x8x256	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, same padding, outputs  8x8x256	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x256  				|
| Fully connected		| (4x4x256) 4096x4096     						|
| RELU					|												|
| Dropout 50%			|												|
| Fully connected		| 4096x4096			     						|
| RELU					|												|
| Dropout 50%			|												|
| Fully connected		| 4096x1024			     						|
| RELU					|												|
| Dropout 50%			|												|
| Fully connected		| 1024x43			     						|
| Softmax				|           									|
 

#### 4. Training the model

##### 4.1 Choosing hyperparameters

Training the model was a series of iterative steps, by repeatedly observing the relative accuracy and total loss between the validation and training data. If the training loss/accuracy goes down but validation loss/accuracy doesn't, then we consider our model as having a high-variance and as a result is overfitting the training data. For example, under the `LeNet` framework, using a `L2 Regularization = 0.1`, we see that the accuracy doesn't improve over the epochs, this shows a high-bias in that our model does not capture relevant features of the data. **Note that the loss curve between training and validation should only be compared in their slope trend but not their magnitude. There are much less validation data than training data and hence their total loss would be much lower than the total loss of the training curve.**

![alt text][image6]

This led us to retrain the network with a lower `L2 Regularization = 0.01`, which gives us a much better fit.

![alt text][image7]

Similarly for learning rate, whenever we notice that the learning plateaus, we rerun the network with a lower learning rate. Usually this results in a better performance and the rate where the improvement stops is the rate we select.

##### 4.2 Choosing model architecture

For the task of classifying traffic signs, we mainly investigated three models. In this report, we shall refer to them as

1. LeNet
2. LeNet2
3. VGG6

Our intention was to start using a simple model and to progressively explore more sophisticated model. For our first model, we based it off the original [LeNet](#model-1-lenet) model with the same number of layers (2 convolutional and 3 fully connected layers). As this is one of the earliest convolutional neural network introduced (1998), this should serve as a benchmark against our more sophisticated model.

Next we attempted a 2-stage model, inspired by a paper by Sermanet, and LeCun working on the same task of classifying German traffic signs. One of the key ideas presented in the paper is the use of connections that skip layers. This simply means that instead of connecting the neural network layers in a linear fashion with each layer's outputs feeding into the next layer as inputs, some of the layers are directly connected to deeper layers. In order to fairly study the impact of skipping layers, we used our first model and simply added an extra connection between the second convolutional layer with the second last fully connected layer. There are no new layers introduced. This model is referred to as [LeNet2](#model-2-lenet2) in this report (see the [Concat Row](#model-2-lenet2)).

Finally, we decided to explore a deeper network. We chose to use a modified version of Simonyan, and Zisserman's VGG network. This is configuration B of ConvNet Configuration from "Very Deep Convolutional Networks for Large-Scale Image Recognition". To reflect the fact that our images are 32x32 instead of 224*224, we removed the last four convolutional and maxpooling layers from their configuration B. As this results in a VGG network with 6 convolutional layers, we refer to this model as [VGG6](#model-3-vgg6). 

Here we show the validation accuracy/loss of the three models we discussed. 

| Model         		|  Accuracy	| 	Loss   |   Time  |
|:---------------------:|:---------:|:--------:| :------:|
| LeNet         		|  94.85%   |  1481.64 |  365.34 |
| LeNet2            	|  95.03%   |  1416.71 |  362.44 |
| VGG6      	      	|  98.46%   |   999.69 | 3106.30 |

Each were trained on the same set of data over the following hyperparameters.

1. Epochs            = 40
2. L2Regularization  = 0.001
3. Optimizer         = Adam
4. Learning rate     = 0.0005
5. Dropout prob      = 0.5

The parameters were selected according to [section 4.1](#41-choosing-hyperparameters). Adam optimizer was selected because we wanted our learning rate to reflect the trend of optimization path. 

#### Commentary

From the results above, we were not convinced that skipping connection (going from `LeNet` to `LeNet2`) resulted in material improvement. We suspect that it might be due to the simplistic implementation of the connection skip we employed. From Szegedy, Liu, et al "Going Deeper with Convolutions", perhaps the layers connected using skipping should undergo some form of transformation (e.g., activation, maxpooling, or convolution) before being fed into other layers to improve model accuracy performance.

On the other hand, it was very clear that the deeper `VGG6` model outperforms both `LeNet` and `LeNet2`. This indicated that deeper networks were significant in improving classification accuracy. Nevertheless, the main drawback was the 8 to 9 fold increase in calculation time.

Given the performance in validation accuracy, we chose `VGG6` as our final model.

#### Results

Below is the plot of training and validation loss on the `VGG6` network

![alt text][image8]

The model achieved a
* validation set accuracy of 98.46%
* test set accuracy of 99.73%


#### 5. Testing Final Model on New Images

We found 5 German Traffic Signs\* from the web, which are then cropped and resized to 32x32 as:
![alt text][image9]

**See ./examples/NewGerman/References.txt for image credits**

\* It was later found out that one of the images was not a German Traffic Sign. But we kept it there because it served to illustrate interesting properties of our model's prediction.

Of the five images, the first one and the last one might be harder to classify. 
The first one is not part of a German Traffic Sign and hence was never present in our training dataset. An example of a German's bumpy road sign is 

![alt text][image10]

which is different from the sign we found in two aspects, 1) bump size is different, 2) background color of yellow instead of white.

On the other hand, the last image is taken at an angle instead of front facing.

The truth labels of those five new images are:

| Image			        |  Label | 
|:---------------------:|:------:| 
| Bumpy road       		|   22 	 | 
| Speed limit (70km/h) 	|    4 	 |
| Right-of-way next		|   11   |
| Road work	      		|   25	 |
| Speed limit (70km/h)	|    0   |

The prediction obtained from our final model are:

| Image			        |  Label | 
|:---------------------:|:------:| 
| Bumpy road       		|   14 	 | 
| Speed limit (70km/h) 	|    4 	 |
| Right-of-way next		|   11   |
| Road work	      		|   25	 |
| Speed limit (70km/h)	|    0   |

We achieved an accuracy of 80%. Indeed, we observed that the 1st image of bumpy road was misclassified. This shows that our neural network model is not general enough to be used on signs that were never trained before (even though as humans, we can tell that both version of Bumpy road logo refer to the same thing).

To further understand how certain our models are in its predictions, we show the top 5 softmax probabilities. (See `GetTopKSoftmax`)

`
22 : Bumpy road
26.42% for 14: Stop
24.32% for 17: No entry
20.75% for 26: Traffic signals
 6.12% for 25: Road work
 4.21% for 22: Bumpy road

4 : Speed limit (70km/h)
100.00% for 4: Speed limit (70km/h)
 0.00% for 24: Road narrows on the right
 0.00% for 26: Traffic signals
 0.00% for 0: Speed limit (20km/h)
 0.00% for 1: Speed limit (30km/h)

11 : Right-of-way at the next intersection
99.99% for 11: Right-of-way at the next intersection
 0.00% for 42: End of no passing by vehicles over 3.5 metric tons
 0.00% for 30: Beware of ice/snow
 0.00% for 6: End of speed limit (80km/h)
 0.00% for 27: Pedestrians

25 : Road work
99.88% for 25: Road work
 0.11% for 33: Turn right ahead
 0.01% for 26: Traffic signals
 0.00% for 30: Beware of ice/snow
 0.00% for 29: Bicycles crossing

0 : Speed limit (20km/h)
99.96% for 0: Speed limit (20km/h)
 0.01% for 37: Go straight or left
 0.01% for 35: Ahead only
 0.00% for 23: Slippery road
 0.00% for 28: Children crossing
`

Again, we notice that our model is quite uncertain in its prediction of Bumpy road because it is a traffic sign it has never been trained on before. Giving us only a 26.42% in the top softmax probability.

#### Summary

We reiterate the key findings of this project in the following bullets
1. We did not find material improvement in skipping connections
2. We did find material improvement in using deeper convolutional networks
3. Our `VGG6` model achieved a 98.46% and 99.73% accuracy on training and validation dataset respectively
4. When applied to new images on signs that it had trained on, our model were very certain and are correct in its prediction.
5. When applied to new images on sign that it had never trained on, our model were both uncertain and incorrect in its prediction.


