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

All the week1 labs are captured at: (With solution,pulls stock data and predict next day trend (up,stay,down)

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
https://screenshot.googleplex.com/BqqXhZMKLRUMXRt

Hard to maintain long term dependency. (due to Vanishing or exploding gradient). None of the prevous techniques like RELU, random weight init (xiavior etc) , L1,L2, gradient clipping but none of these mitigate the vanishing gradient. Architecure solution fix the problem (LSTM...)

https://screenshot.googleplex.com/7Kkhr7WLJpmycFN

# LSTM , GRU













