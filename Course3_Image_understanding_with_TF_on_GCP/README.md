### Week 1

## Linear model for image classification

![image](https://user-images.githubusercontent.com/1594001/147507868-ba729d64-4705-4b78-b124-fc6321d2e97e.png)

#### prediction 

![image](https://user-images.githubusercontent.com/1594001/147507972-e48c30a1-1d4a-46d8-883e-a99ffbb0d41f.png)

### Linear model using tensorflow.

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/08_image/mnist_linear.ipynb


# DNN , dropout , CNN using tensorflow

Talks about local training, training in cloud (using docker image) and deployment.

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/image_classification/solutions/2_mnist_models.ipynb


# Week 2
How to deal with data scarcity?

A) Agument data using image transformation functions

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/08_image/flowers_fromscratch.ipynb

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/image_classification/solutions/5_fashion_mnist_class.ipynb

OR

B) Use transfer learning:

In a Neural network last few (dense) layers are task dependent. So in transfer learing cut them off and retrain.

Video going through the transfer learning in GCP: Use dataflow api to preprocess data (read csv, do preprocessing and write tfrecord for faster read by subsequent steps). Use ml-engine or ai-platform for training and test. All commands to train and test are listed in video.

https://www.coursera.org/learn/image-understanding-tensorflow-gcp/lecture/m2mfk/lab-solution-implementing-transfer-learning




Deeper network problem
1) Covarince shift. During the training passes change in earlier layers weights continously shift the weights of later layer. This can be mitigated by adding batch normalization. Can help you train NN upto 20 layers.
2) Vanishing gradients: Resnet. Bypass connection (shortcut connections) to the forward layer. Can train > 50 layers.

Use TPU: Resnet (50 layers) can full run in 30 mins in TPU.
How to make your code run on TPU?

![image](https://user-images.githubusercontent.com/1594001/147537950-2fc13a89-d72a-44c2-8753-ff7869162f33.png)

Use TPU only for training. Works well if you have batch of inputs. Not of predicting single case.
Use tpu flag to false for inference.
TPU have large matrix multiplication.
TPU use bfloat(16 bit) or float32. Usecases which requires double precision airthematic is not good fit

TPU related changes in a video.
https://www.coursera.org/learn/image-understanding-tensorflow-gcp/lecture/HGt7u/demo-tpu-estimator


https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/08_image/flowers_fromscratch_tpu.ipynb


# Neural architechral search.

Can you automate the building of model?

10 layer will have 10B networks.
Use reinforcement learning to do neural architechral search.

![image](https://user-images.githubusercontent.com/1594001/147538810-a5599da9-7779-4b93-b1b7-e4b4a0370afc.png)

Forms the basis of Auto ML. 

![image](https://user-images.githubusercontent.com/1594001/147540735-206316c1-a837-4a5d-91fe-2eb9fbb519b0.png)

### Codeless models

Cloud Vision API
-----

cloud.google.com/vision out of the box label (pre trained). You can't give custom label. etc. For custom label use Auto ML below.

Auto ML Vision API. (Uses Neural architechural search)
------
Upload your own data and train (incremental train) the model.
define your label.
Upload photos and add add labels.




















