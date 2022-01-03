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

Synthetic data (sine wave with varying amp,frequency,noise) results:
-----

![image](https://user-images.githubusercontent.com/1594001/147974831-31157448-723d-41e7-b85f-dd62633ea86e.png)

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/09_sequence/sinemodel/model.py

Video of training linear model in GCP. (includes commands)

https://www.coursera.org/learn/sequence-models-tensorflow-gcp/lecture/5jdJd/lab-solution-using-linear-models-for-sequences

ARMA model (still linear model):

Model the difference between the moving averages and future values.

What if the relationship is non linear?  Use DNN. (multilayer + non linear activation function)



