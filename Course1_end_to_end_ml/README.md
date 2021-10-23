General ML pipelines
---------------------

This course teaches full ML development flow using 7 Qwick labs listed as follows.

![image](https://user-images.githubusercontent.com/1594001/137011359-33034e95-178c-48d6-a2ec-99fa36979c90.png)

![image](https://user-images.githubusercontent.com/1594001/137022365-a430a151-7c45-4c04-ba8d-7dcdbb1a8138.png)


To effectively do ML you might need a GCP ML tool (Cloud ML Engine). This course use tf.estimater level.


![image](https://user-images.githubusercontent.com/1594001/137011595-6daa6d28-0253-4fcf-a591-2214152689cf.png)


Large datasets will not fit in memory of single machine. For good ML results your final pipeline should not sample data or aggregrate data.

Model training in large dataset is not a emberassing parallel nature. Gradient decent are not embarassingly parallel you need parameter servers which have shared memory to update the parameter during each epoch. Scaling out is the answer (not scaling up).

Big data ---> feature engineer in full data set --> Use appropriate model architecture (CNN / RNN etc)---> Deplay and autoscale prediction code. ----> REST API. (query/seconds)

Hyperparameter turning also gives a decent improvements.

Who does the preprocessing, encoding (embedding) scaling during prediction phase? Avoid Training servering skew. Use Cloud ML engine.
tf.transform for precessoing/encoding, scaling.

Cloud data flow (for preprocessing) and Cloud ML engine are both serverless techonology which scales for very large datasets!

**Explore the data:**
Gives you sense of data, what it looks like, what is its distrubiton, missing values etc. Develop the understanding of data.
During production you have to deal with data as it comes.

Most model are operated in structured data, ie MLP (~60%).LSTM (~30).CNN (~5%)

High level flow::
------------------

Use jyupter notebook in cloud to do single machine training:
Understand your data (Exploratory data analysis) ----> Sample (subset) the data ----> Data cleaning ----> train a model in Single VM ---> SCALING STEPS

SCALING STEPS: Use apache beam to do transformation (Google Dataflow)  -----> Use Cloud AI (distributed training) -----> Model serving.

Module #1 
--------------
- Structure data (table). predicting weight of baby right after Birth.
- EDA
    - Explore data: Find out which field (columns) have impact on target variable(mean). So if you have X1,X2,....Xn --> Y figure out which of X1,X2...Xn different value have different mean in Y. If sex(male/female) does not change the mean of Y , it is highly likely its not a good predictor (feature) for Y. You can also figure out how many samples you have for each column value of (X1,X2....Xn) This will help you in figuring out how bananced your dataset is.
    - GCP your data can be usually in bigquery (data warehouse) or GCP storage. Notebook architecure. https://www.coursera.org/learn/end-to-end-ml-tensorflow-gcp/supplement/875zO/ai-platform-notebooks    
    - How to create a Notebook? In GCP console (console.cloud.google.com) GOTO "AI Platform" ====> Notebooks ==> Enable Notebook API ==> Create Instance. This will take 2-3 mins to launch a notebook session.(Create a VM for it) ==> Open Jupyter Lab.
    - You can create a terminal and clone repo etc in it.
    - Key idea is to use group by on a column to generate the count and avg and plot it to know if your variable have any impact in the target var. 
    - Big query in this case was able to aggregrate/group by query for ~30 M record in less than 3 sec.

Sample big query.

```python
# Create SQL query using natality data after the year 2000
query = """
SELECT
  weight_pounds,
  is_male,
  mother_age,
  plurality,
  gestation_weeks,
  FARM_FINGERPRINT(CONCAT(CAST(YEAR AS STRING), CAST(month AS STRING))) AS hashmonth
FROM
  publicdata.samples.natality
WHERE year > 2000
"""

# Call BigQuery and examine in dataframe
from google.cloud import bigquery
df = bigquery.Client().query(query + " LIMIT 100").to_dataframe()
df.head()

# Create function that finds the number of records and the average weight for each value of the chosen column
def get_distinct_values(column_name):
  sql = """
SELECT
  {0},
  COUNT(1) AS num_babies,
  AVG(weight_pounds) AS avg_wt
FROM
  publicdata.samples.natality
WHERE
  year > 2000
GROUP BY
  {0}
  """.format(column_name)
  return bigquery.Client().query(sql).to_dataframe()
  
# Bar plot to see is_male with avg_wt linear and num_babies logarithmic
df = get_distinct_values('is_male')
df.plot(x='is_male', y='num_babies', kind='bar');
df.plot(x='is_male', y='avg_wt', kind='bar');


# Line plots to see mother_age with avg_wt linear and num_babies logarithmic
df = get_distinct_values('mother_age')
df = df.sort_values('mother_age')
df.plot(x='mother_age', y='num_babies');
df.plot(x='mother_age', y='avg_wt');
```
**Full Lab 1 location for exploratory data analysis is**
https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/06_structured/1_explore.ipynb

Big query tutorial https://cloud.google.com/blog/topics/public-datasets/google-bigquery-public-datasets-now-include-stack-overflow-q-a

Module #2
---------
Prototype a model fast by sampling data (creating small dataset). Then scale it out by using full dataset later.

Sampling data, data preprocessing (encoding), building model.

1. Speed up training iterations during development. Sampling the big data to make it smaller and develop your model on smaller set ( make sure it is representative of full dataset). START OUT SIMPLE and iterate fast in small dataset. Make sure your splitting statergy does not have data leakage. Look for nearlyl identical rows in datasets whcih can go to train or test during split. Using FARM_FINGERPRINT is one of the way to hash and avoid this data leakage.

Let say if few varable may or may not be available at prediction time? how do you deal with it.
Approach #1: Build two model , one with those variable in the data, other without those variable.
Approach #2: Build only one model but train in both fully known and masked data.

2. If you know that during predict phase some feature might or might not be available, you should make a copy of current data, mask the missing var and append it to the orginal data. This will create a big dataset which when trained will not be dependent on optional variable.

4. Data preprocessing is captured in notebook below. Which include removing missing value rows. Use FARM_FINGERPRINT, ABS,MOD function to sample data randomly.

Lab2: (Data sampling, dataset creation)

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/end_to_end_ml/labs/sample_babyweight.ipynb

More about tf.estimator

![image](https://user-images.githubusercontent.com/1594001/137040801-5a7d581e-013e-489c-8f55-e96fde02f24f.png)

How to create a encoding and create feature_column

![image](https://user-images.githubusercontent.com/1594001/138500694-b180eaca-1357-4c3f-aea1-3e24afe49631.png)

Consider DNN when you have "dense features" (images, pixels in range 0-255) and correlated features (nearby pixels in images)
Consider linear model for sparse independent features. 
DNN works well for dense highly correlated features. DNN does not work in sparse data.

Used "deep-and-wide model" (tf.estimator.DNNLinearCombinedClassifier) if you have both dense and sparse feature (eg. employee id). Sparse input --> connect them directly to output , dense input--> multiple layers.

If both use "wide and deep model." : takes sparse input and connect them directly to output. and pass dense input through multiple layer. DNNLinearCombinedClassifier

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/end_to_end_ml/solutions/keras_dnn_babyweight.ipynb

Look into the categorical_fc fucntion to see how to encode variable.
Use tf.estimator.* API for training model. training loop save the model in checkpoint. eval loop restore the model and evaluate 

Use tf.estimator.train_and_evaluate for continuous training and evaluating.

Module 3
---------------
For full dataset we want to use "Distributed training and model serving". Define apache beam pipeline, execute in cloud dataflow

Apache beam: data processing pipeline. (support both batch and steram). SDK. you can define your own pipeline.

Cloud DataFlow: Serverless, fully managed for data preprocessing. 

Define Apache beam pipleine:
read(from gs) --> map(),group(),flatmap ---> write to gs. 
beam have many io connector.

sort of pipeline in beam. Have many connector for different sources (bigquery,file etc) Cloud dataflow: execute the code written in apache beam.serverless fully managed service support java/python. 

            cloud dataflow
            
Bigquery -----------------------> cloud storage 


![image](https://user-images.githubusercontent.com/1594001/132077628-31cd8148-3f11-410f-a452-bb41752ec5b5.png)

![image](https://user-images.githubusercontent.com/1594001/132077609-bfd018c4-94db-46fc-99a4-cd4b2514c28c.png)

How to run the pipeline

![image](https://user-images.githubusercontent.com/1594001/132077689-a5aa8e4d-131e-4913-8e50-757775ec41bf.png)

beam and dataflow pipeline example notebook:

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/end_to_end_ml/solutions/preproc.ipynb

once you run the beam pipeline in cloud dataflow, it will generate two csv (train.csv,test.csv). These are prepossed and contain the entire dataset.

Training in cloud (CMLE)
------------------------
Do training using "cloud ml engine" : CMLE is execution environment for ml jobs.

Two options:
1. Package your file in python module. 
2. Or make a container (docker). Making docker is bit more complex but gives you ways in which you can manage your own (ai) package dependencies.

Making python package:
task.py : code to parse command line args. calls the code in model.py
model.py: tensorflow code. Should have all code for train_and_evaluate fucntion as well as serving_input_fn

Note: The tensor model should be in package before running (train or prediction) in CMLE "cloud ml engine"

At the end of training the model will be dumped to google cloud storage. The serving function need to be specified during export.(exporter parameters)
tf.estimator.LatestExporter() create exporter object.

First train locally in small subset use --train_example=<small_number> to make sure setup is corect, than in cloud to speed up training process.

Local training: package trainer.task
python -m trainer.task (parameters) // Train locally in standalone mode

// TRAIN IN CLOUD (Batch Job Launch)
gcloud ai-platfrom jobs submit training $JOBNAME --module=trainer.task (otherargs)

// Batch prediction.
gcloud ai-platform jobs submit prediction

// PREDICT IN CLOUD (locally)
gcloud ai-platform local predict --model-dir=$MODEL_LOCATION --json-instances=inputs.json

![image](https://user-images.githubusercontent.com/1594001/138511249-6a62e21b-179b-43a1-bb23-ba72e81370db.png)

![image](https://user-images.githubusercontent.com/1594001/138511167-791ff584-08f6-4b5e-8f34-ef512ee92aaa.png)

Make all the configuration and (hyperparams) as command line args, so that hyperparameter tuning is easy later.

Use tensorboard to monitor the training job make sure your 
- loss converges
- eval does not goes up
- layers does not die.

Training locally and in cloud notebook. (Use python module or docker.)

Kera model training:

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/end_to_end_ml/solutions/train_keras_ai_platform_babyweight.ipynb

tf model training: (much similer than kera's model but concept are similar)

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/06_structured/5_train.ipynb


Deploy in cloud (Create REST end point) 
----------------------

Notice you can't use the train function for serving as for serving the input can be different (json not csv), their might be smaller subset of features. Also the labels will not be there in the serving function.
create a serving function with preprocessing logic (encoding etc). serving_input_fn()

Deployment of model in Cloud AI Platfrom: All you need is the location of your trained model director.

In this case you will be able to deploy and serve model. REST endpoint is created in google cloud. You need to do two commands:

gcloud ai-platform models create ${MODEL_NAME} --regions ${REGION}

gcloud ai-platform versions create {MODEL_VERSION} --model ${MODEL_NAME} --orgin {MODEL_LOCATION}

Go to gcp console and check your model deployed in UI.

use 

api = discovery.build()

REST endpoint 
api = "https://ml.googleapis.com/v1/projects/{}/models/{}/versions/{}:predict" \
         .format(PROJECT, MODEL_NAME, MODEL_VERSION)

See the notebook for model deployment and serving as rest endpoint

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/end_to_end_ml/solutions/deploy_keras_ai_platform_babyweight.ipynb

TF deploy simple 2 command .

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/06_structured/6_deploy.ipynb


flask + app engine to deploy model and get back prediction. REST API

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/end_to_end_ml/solutions/serving_babyweight.ipynb


