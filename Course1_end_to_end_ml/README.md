General ML pipelines
---------------------

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
    - GCP your data can be usually in bigquery (data warehouse) or GCP storage https://www.coursera.org/learn/end-to-end-ml-tensorflow-gcp/supplement/875zO/ai-platform-notebooks       - How to create a Notebook? In GCP console (console.cloud.google.com) GOTO "AI Platform" ====> Notebooks ==> Enable Notebook API ==> Create Instance. This will take 2-3 mins to launch a notebook session.(Create a VM for it) ==> Open Jupyter Lab.
    - You can create a terminal and clone repo etc in it.
    - Key idea is to use group by on a column to generate the count and avg and plot it to know if your var have any impact in the target var. 
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

Full Lab location for exploratory data analysis is 

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/06_structured/1_explore.ipynb


Big query tutorial https://cloud.google.com/blog/topics/public-datasets/google-bigquery-public-datasets-now-include-stack-overflow-q-a

Module #2
---------
Sampling and data preprocessing.

1. Speed up training iterations during development. Sampling the big data to make it smaller and develop your model on smaller set ( make sure it is representative of full dataset). START OUT SIMPLE and iterate fast in small dataset. Make sure your splitting statergy does not have data leakage. Look for nearlyl identical rows in datasets whcih can go to train or test during split. Using FARM_FINGERPRINT is one of the way to hash and avoid this data leakage.

2. If you know that during predict phase some feature might or might not be available, you should make a copy of current data, mask the missing var and append it to the orginal data. This will create a big dataset which when trained will not be dependent on optional variable.

4. Data preprocessing is captured in notebook below. Which include removing missing value rows. Use FARM_FINGERPRINT, ABS,MOD function to sample data randomly.

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/end_to_end_ml/labs/sample_babyweight.ipynb

Consider DNN when you have dense features (images, pixels in range 0-255) and correlated features (nearby pixels in images)
Consider linear model for sparse independent features. DNN for dense correlated features.

If both use "wide and deep model." : take sparse input and connect them directly to output. and pass dense input through multiple layer. DNNLinearCombinedClassifier

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/end_to_end_ml/solutions/keras_dnn_babyweight.ipynb

Look into the categorical_fc fucntion to see how to encode variable.
Use tf.estimator.* API for training model. training loop save the model in checkpoint. eval loop restore the model and evaluate 


Module 3
---------------
Distributed training and model serving

Apache beam: data processing pipeline. (support both batch and steram). SDK. you can write 

read(from gs) --> map(),group(),flatmap ---> write to gs. 

sort of pipeline in beam. Have many connector for different sources (bigquery,file etc) Cloud dataflow: execute the code written in apache beam.serverless fully managed service support java/python. 

            cloud dataflow
            
Bigquery -----------------------> cloud storage 


![image](https://user-images.githubusercontent.com/1594001/132077628-31cd8148-3f11-410f-a452-bb41752ec5b5.png)

![image](https://user-images.githubusercontent.com/1594001/132077609-bfd018c4-94db-46fc-99a4-cd4b2514c28c.png)

How to run the pipeline

![image](https://user-images.githubusercontent.com/1594001/132077689-a5aa8e4d-131e-4913-8e50-757775ec41bf.png)



