#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[imageviz]: ./writeup/visualization.png "Visualization"
[imageequalized]: ./writeup/equalized.png "Grayscaling"
[image3]: ./writeup/random_noise.jpg "Random Noise"
[image4]: ./internet-signs/100.jpg "Traffic Sign 1"
[image5]: ./internet-signs/stopschild.jpg "Traffic Sign 2"
[image6]: ./internet-signs/verbot-der-einfahrt.jpg "Traffic Sign 3"
[image7]: ./internet-signs/vorbeifahrt-rechts.jpg "Traffic Sign 4"
[image8]: ./internet-signs/vorfahrt.jpg "Traffic Sign 5"
[dropoutpaper]: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/stela/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set.

I used the python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3 (width, height, color-channels) 
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.
It is a set of randomly chosen images in the training data set.

![alt text][imageviz]

As you can see the brightness and (I believe) contrast differ quite a bit among the images.
To simplify the job of the neural network, equalizing the images was part of preprocessing.
The visualization doesn't explore differences in quantities per type of traffic sign,
I got decent results without trying to take that into account, one way or the other.
If the distribution of signs in the testing data are typical of that at the roadside,
it might simply be a good idea to use the input data as-is, to make sure that at least
common signs are accurately recognized. 

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it was hinted to do so,
and people in the slack forums didn't seem to report much negative effects of doing so,
and last but not least it should speed up processing speed if data is reduced by a third.
I weighed the RGB-channels according to a formula I found on the net, which is supposed to be
similar to how the human eye does it.

I then divided the pixel values, now ranging roughly 0-255, by 256,
which gave a new range of 0.0-1.0. This helped to improve training accuracy quite a bit.

Then I used skimage.exposure.equalize_adapthist() to equalize the images.
On first attempt it failed pretty horribly (NaN pixel values and too dark images),
but after making sure the pixel values were non-negative
(I initially tried to center the values around zero _before_ instead of after equalizing),
equalize_adapthist() worked well. Afterwards I subtracted 0.5 from each pixel,
which again led to improved accuracy.

Here is an example of traffic signs after grayscaling and equalization.

![alt text][imageequalized]

I decided not to augment/generate additional data because I got good enough accuracy
without it, and augmenting seemed fairly labor-intensive. Instead of adding
data I took other measures to avoid overfitting: dropout and
clipping the weights.  


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers,
very similar to the LeNet architecture but with the following adaptations:
* 43 classes instead of 10
* greyscale input
* Use of dropout layer while training
  
| Layer         		|     Description	        						| 
|:---------------------:|:-------------------------------------------------:| 
| Input         		| 32x32x1 greyscale image   						|
| Convolution 5x5     	| 1x1x1 stride, valid padding, outputs 28x28x6	    |
| RELU					|													|
| Max pooling	      	| 2x2x1 stride, outputs 14x14x6						|
| Convolution 5x5     	| 1x1x1 stride, valid padding, outputs 10x10x16	    |
| RELU					|													|
| Max pooling	      	| 2x2x1 stride, outputs 5x5x16						|
| Flatten               | 5x5x16 -> 400                                     |
| Fully connected		| 400 -> 120        								|
| RELU					|													|
| Dropout               | 50% dropout when training, 0% when validating     |
| Fully connected		| 120 -> 84         								|
| RELU					|													|
| Dropout               | 50% dropout when training, 0% when validating     |
| Fully connected		| 84 -> 43           								|
| Softmax				|           										|
 
I didn't tweak the parameters of the network further than functionally necessary,
since it would be time-consuming and I was satisfied with the accuracy
when also doing preprocessing + dropout + weight-clipping.  

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the modified LeNet network (see above)
with an Adam optimizer, since it was supposed to be one of the better ones.
Since the Adam optimizer adapts it learning rate,
there should be no point in trying to artificially decay the learning rate as can be done with other optimizers.
According to [Dropout: A Simple Way to Prevent Neural Networks from Overfitting][dropoutpaper], see e.g. table 3
of that paper, dropout is superior to L2 regularization, so I used dropout instead.
I experimented with higher or lower learning rate, but 0.001 worked well.
After adding dropout to a second layer, training required more epochs but gave
better long-term results, probably through reduced overfitting, at least 50 epochs were useful. 
The batch size of 128 from the LeNet assignment seemed to work fine, so kept it.   

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 96.6% 
* test set accuracy of 95.3%

The high training set accuracy indicates over overfitting.
If there was a need to improve accuracy further I'd try augmenting the test data first.  

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


