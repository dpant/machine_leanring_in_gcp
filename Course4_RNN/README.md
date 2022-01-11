# week 1

*Sequence model:*

When can i apply sequence model?

1. Data is timeseries, changes over time
2. Check after doing slice (windowing) and see if you can predict next observation using the current window

types of sequence model (RNN)

One-to-sequence: poetry generation

sequence-to-one: smart reply

sequence-to-sequence: Machine translation

Sequence data may have: trend + seasonality (cyclic)

load csv (data is one window per row) --> ML Model ( linear/dnn/cnn/rnn/lstm)

All the week1 labs are captured at: (With solution,get stock data from bigquery and predict next day trend (up,stay,down)

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/09_sequence/sinewaves.ipynb


Synthetic data (sine wave with varying amp,frequency,noise) results:
-----

![image](https://user-images.githubusercontent.com/1594001/147974831-31157448-723d-41e7-b85f-dd62633ea86e.png)

In the above dataset there is not much of improvement by using complex model. Maybe the relationship in the data is not complex (or capturable) by complex models.

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/09_sequence/sinemodel/model.py

Video of training linear model in GCP. (includes commands)

https://www.coursera.org/learn/sequence-models-tensorflow-gcp/lecture/5jdJd/lab-solution-using-linear-models-for-sequences

ARMA model (still linear model):

Model the difference between the moving averages and future values.

What if the relationship is non linear?  Use DNN. (multilayer + non linear activation function)

#### DNN
In the dataset provided DNN do provide some better performance.


### CNN

What Images and sequences have in common? locality!!!
Pixels. 

How convolution work in 1-D. 

Which filter is more specific? longer filter are more specific.

![image](https://user-images.githubusercontent.com/1594001/147989196-d13660ae-81d3-450c-b860-4c0109836415.png)

![image](https://user-images.githubusercontent.com/1594001/147989382-c1cf5cbf-be7f-4f1e-95e1-078917ac5dae.png)


CNN performed worst than DNN in this case. Why? variable length problem.

## variable length problem. (CNN/DNN)

Sequence domain model should be robust for the length of the pattern.

handling variable length input or output

How can you solve variable length?

Cutting and padding:

- While training feed fixed size (windowed) inputs.
- Let say during prediction you just have less the window size of datapoints? you will pad it. This will loose some of the weights during prediction (set them to 0) 
- You can potentially workaround this by using smallest size window you will see in prediction time. But this will again limit your model capability as it is seeing smaller window at a time.

Bagging 
Take avg of each characterstics. "bags of words"
Throws away order. The cat sat on the mat === the mat sat on the cat.

### RNN
Handles variable length sequence.

![image](https://user-images.githubusercontent.com/1594001/148111769-ad4d6267-a043-4c96-8ad7-8f28d26a0e95.png)

RNN is good sequence feature extractor.
RNN compact previous input.
RNN output is not just a class label, it is a sequence.

![image](https://user-images.githubusercontent.com/1594001/148135434-cacc5d4d-87fb-4479-bc78-5052521880b2.png)


Hard to maintain long term dependency. (due to Vanishing or exploding gradient). None of the prevous techniques like RELU, random weight init (xiavior etc) , L1,L2, gradient clipping but none of these mitigate the vanishing gradient. Architecure solution fix the problem (LSTM...)

![image](https://user-images.githubusercontent.com/1594001/148135462-6c65b79c-6167-4521-a473-f2759b7e5ec8.png)


RNN ==> Single cell , single set of weights. Max sequence length of RNN (unrolling size of cell)

Michel C was born in paris , France. His mother tounge is ???
How will RNN leanrn it? RNN learn dependency upto its max sequence length. 

![image](https://user-images.githubusercontent.com/1594001/148121956-c25066e1-51ff-4b28-b1d4-4e6505848846.png)

for long term dependencies you have to increase the max seq length (T) which will make the RNN training harder. Think T=50 as a 50 layer deep neural network. It will have vanishing gradient problem.


# LSTM , GRU
Dealing with deeper sequences.

Hidden state vector (H), can be also deemed as output.

![image](https://user-images.githubusercontent.com/1594001/148122274-aa6e09d2-e456-4fb0-853d-098ffaa7aa76.png)


LSTM have another state vector is C (cell state). Similar to shortcut connection in Resnet.
Three states:
What to forget from cell state, what to update the new data , what data from cell state to expose to hiddent state.

![image](https://user-images.githubusercontent.com/1594001/148122386-f9e64917-0666-40bb-82ad-7d0caa0316a6.png)

![image](https://user-images.githubusercontent.com/1594001/148123181-efdd72cf-7def-4b61-b920-4e2e67d30a65.png)


# GRU
More computational efficient than LSTM 

![image](https://user-images.githubusercontent.com/1594001/148123264-3a275e4d-e3d8-4bd0-a230-662e1377cfb5.png)

Two ways to do RNN in tensorflow

tf.keras or tf.nn

![image](https://user-images.githubusercontent.com/1594001/148123639-47229b53-1869-4e34-a51f-cdffe3ffb3a0.png)

![image](https://user-images.githubusercontent.com/1594001/148124918-8361ddf4-e4d7-46e6-9af7-f9e54c90150e.png)

Video talking about RNN and dimension check for debugging.

https://www.coursera.org/learn/sequence-models-tensorflow-gcp/lecture/ztHaR/lab-solution-time-series-prediction-end-to-end-rnn


# Deep RNN

![image](https://user-images.githubusercontent.com/1594001/148135995-cde3506c-977f-4430-88d6-926d36809afa.png)

![image](https://user-images.githubusercontent.com/1594001/148136342-073f2b64-e8ac-484e-96cb-c3122acc1b3b.png)


Deep RNN takes significat time to converge.

# improving loss function using intermediate steps of RNN

Use output value to calculate loss function from intermediate steps also.

Do average loss over multiple prediction "K back steps of RNN" to feed in subsequent DNN layers.

The following is using all the outputs of RNN (for every timestep)

h1 = tf.layers.dense(**outputs**, cells.output_size, activation=tf.nn.relu)
h2 = tf.layers.dense(h1, cells.output_size // 2, activation=tf.nn.relu)

You beed to modify the loss function to take care of only last "K" states
aka compute_errors in
https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/09_sequence/sinemodel/model.py



# Dealing with real sequence data

Case study Weather data: (long term seasonal pattern in data)
If we have too much data in one sequence split the sequence.
- What is the subsequence length. (Try different values within the resonable bound)
- What is the overlap of the sub-sequence. Overlap give more data.
- if successive sequence in data are continuous and non-overlapping we can have a trick in which we feed the output of last RNN to input of next batch. This gives a effect of training on a longer sequences.

![image](https://user-images.githubusercontent.com/1594001/148137546-e5e93a90-6dca-447c-af09-3b659ad8c5e0.png)

How to predict (N next time step)?
Take the prediction at time "t+1" and feed it back to get the prediction "t+1" (Error will accumulate). You need to pass output as well as the state (which capture all the context) to the next step.

OR use encoder-decoder model.

- Resampling data: Preprocessing step Average  (reading per min --> daily reading Or daily reading --> weekly reading)
- One model vs multiple model: Does weather pattern from one sensor generalize to other? if yes make one model. If no make different models. Combine lat,long in the data feed.
- Add the non sequentail feature (lat/long) to the DNN (after the RNN steps). Don't add constant feature in RNN it may take away their predictive power either by adding noise or hiding the real pattern.

**From project notebook:**

The ML model was able to capture the variation and seasonality in the weather data beautifully. IN RNN there is a lot of hyperparamater tuning needed. Without hyperparameter tuning results might be worthless.

SEQLEN ( RNN unrolled seqlen should be atleast as big a our prediction window size. Increasing SEQLEN makes the model hard to train so resample the input and you can reduce the SEQLEN.
Macro signal vs micro signal.
N_FORWARD: label N step ahead.

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/09_sequence/temperatures.ipynb

## Week2

Text classification. (its a sequence as sequence is important in text)

Eg: Email spam/non -spam. Customer classification, style classification, Predict publisher.

Find numeric representation of words to number with retain as much meaning

![image](https://user-images.githubusercontent.com/1594001/148151360-f72867b5-47db-4e49-b074-a9c1e050aef9.png)

Encode only topK words. including sparse words will confuse the model.
Out of vacab will be encode to 0
Each example in batch should be of same size (efficency reason). So pad to make all the example of same size.

You need embedding to remove the sparsity of the following process.
Text --> number --> one hot (sparse problem)

Embeddeding is learned by dense layer (embedding layer) to our network whose width is much smaller than our one hot encoding . it compress the one hot to floating represenation which capture meaning.

![image](https://user-images.githubusercontent.com/1594001/148152472-4d18eeb3-a92b-4acc-b89f-162bb5657b54.png)

Kera's API is simpler to use than tensforflow and is becoming the defaco for ML model. Keras does not support distributed training. use keras.estimator.model_to_estimator() to convert to tensorflow model.

CNN model are good to analyze group of words. In image classification each pixel is represented by three prixel (R,G,B) ; In text classification each word is represented by the number of dim in embedding space. 







