# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./examples/PilotNet.png "PilotNet"
[image1]: ./examples/model_viz.png "Model Visualization"
[image2_L]: ./examples/Left.png "Normal Driving Left Camera"
[image2_C]: ./examples/Center.png "Normal Driving Center Camera"
[image2_R]: ./examples/Right.png "Normal Driving Right Camera"
[image3]: ./examples/Recover.png "Recovery Image"
[image_YUV]: ./examples/YUV.png "YUV Image"
[image_norm]: ./examples/Norm.png "Normalized Image"
[image_crop]: ./examples/Crop.png "Cropped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
	* read_data.py containing code to load the training data and do a color conversion. Data is saved as .npy arrays at this step which can be loaded in model.py
* drive.py for driving the car in autonomous mode (modified to convert images to YUV)
* model.h5 containing the trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results (this file)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

After reading the [NVIDIA PilotNet Paper] (https://arxiv.org/pdf/1704.07911.pdf) recommended in the course, I decided to start with PilotNet. 
I pre-processed the images by first converting to YUV (as described in the paper), normalized the image, and finally cropped the top 50 pixels and the bottom 25 pixels.
I ensured that the neural net was initialized with normal distributions then started exploring the activation function options.
In order to ensure a variety of non-linearity I wanted to have at least two different activation functions. After testing, I decided on the ELU (expontential linear unit) and tanh (hyperbollic tangent).
The exact architecture of my neural net will be detailed later in this writeup.

#### 2. Attempts to reduce overfitting in the model

I added 3 dropouts to PilotNet to ensure my model would not overfit. 
I also reserved 20% of my training data for a validation set and tried to tune my model such that training error was much lower than validation error (an indication of overfitting).
Finally, I recorded data at various speeds, driving behaviors, and both directions around the track to ensure data diversity and encourage a generalized solution.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

My strategy for generating training data was to record generalized data that covered desired behavior around the track as well as key manuevers and areas that I wanted to enhance the neural nets success at.
Specifically, I always recorded clockwise (CW) and counter-clockwise (CCW) data for any manuever.
I recorded both fast (30 mph) and slow (10 mph) data for my full nominal laps (centered smooth driving).
I then reocrded left and right side recoveries (record the car driving from a lane edge to the center of the lane).
Finally, as I trained my model, I noted there were two tight corners which my neural net would fail at.
I theorized this was due to the disproportionate nature of my training data of many sections with small turn angles and few sections with large turn angles.
I recorded additional successful driving and recovery in those tight turn areas. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to research existing neural nets and leverage those. 
As mentioned before, I read the [NVIDIA PilotNet Paper] (https://arxiv.org/pdf/1704.07911.pdf) recommended in the course, and I decided to start with PilotNet.
PilotNet is clearly appropriate because it was used to also drive a vehicle autonomously from image data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation (20%) set. 
I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added 3 dropout layers and tweaked the keep rate until validation error was not substantially higher than training error (within ~1%).

As I trained my model, I took note of the areas where my vehicle would drive off the road and recorded some additional data at those targeted areas.

I also increased the batch size to 64 to ensure the net had a good sample of the driving maneuvers required by the course.
I noted that after I increased the batch size, my training/validation error would not improve after the first epoch, so I ended up only running one epoch.

I also explored various activation function options and finally settled on the ELU (expontential linear unit) and tanh (hyperbollic tangent).

In the end, my neural net was able to successfully navigate the first track with only slight deviation towards a edge once, but it was able to recover.

#### 2. Final Model Architecture

The final model architecture is a modified version of PilotNet:

![alt text][image0]

I added dropout and changed the activation functions. 
Here is a the architecture I used for my solution:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

The details of my training data collection and process are best summarized in section 4 (Appropriate training data) above.

When training the model, I utilized all three camera images (left, center, and right) with steering correction factors of (0.3, 0, -0.3) respectively.
Sample images of normal loops can be seen below:

![alt text][image2_L]
![alt text][image2_C]
![alt text][image2_R]

My recovery training data looks like this:

![alt text][image3]

After my data collection, I had <<TODO>> sample images. I utilized all three cameras which increased my training size to <<TODO>>.

The first step of my pre-processing pipeline was to convert each image to YUV:

![alt text][image_YUV]

The next step was to normalized the image:

![alt text][image_norm]

Finally, the images were cropped to remove the skyline and the hood of the car:

![alt text][image_crop]

When training my neural net, I made sure to randomly initialize the training parameters with normal distributions.
I utilized the mean square error cost function and the adam optimizer so no learning rate tuning was necessary.
I trained my model with 1 epoch, a batch size of 64 and randomly shuffled the data with a 20% of the training set reserved for validation.
Watching the training error helped me discover that no improvement was made after the first epoch.
The validation error helped me prevent overfitting by tuning the dropout layer keep rates.
Ultimately my net was able to successfully navigate the first track but not the second.
In order to improve the performance on the second track, my next step would be to collect training data from the second track (this would also improve model generalization).
In order to see my final run, please view video.mp4.