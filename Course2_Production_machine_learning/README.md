### Architecting ML system.
ML model account for only 5% of total code base.

keep system running in production need a lot of other pieces.

Serving paradigm, scable serving, design architecure.

Use managed services to save configration time: Dataproc, AI platform, DataFlow

General pipeline.

0. First Define business case and success cretria.
1. Data extraction (ingestion) streaming/batch.
2. Data analysis (EDA): sample stats, outlier,trend and distribution. Identify features. Distribution on input change can indicate something bad in encoding (wrong encoding)
3. Data prepration: data transformation, feature engineering. Encode categorical variable. 
4. Choose and train a Model. Feed ML model with data, Model estimate (Out of sample accuracy), 
5. Model validation: notify engineer (alerts) to when accuracy of a region goes down.
6. Push model to to model regitry. (who trained and published model, which data used, when deployed)

![image](https://user-images.githubusercontent.com/1594001/138536318-1c27c9c2-04ed-4d1f-865d-c437f1c8fcc9.png)


Use AI platform prediction service.


![image](https://user-images.githubusercontent.com/1594001/138536357-2252b870-73d7-4563-aa63-0b7f5e6a37b3.png)


Training design decision: (static vs dynamic)
----------

Is the thing you are modeling are fixed (like physics rules) or every changing (like fashion)?

Static training vs Dynamic training

Static training: train once and pushed to AI platform. refresh manually once in a while. 
Dynamic training: continously training as data comes in. Use cloud function, App engine, and dataflow to ingest new data and trigger re-training.

For dynamic traning, you need more monitoriing, rollback, data quarunteen capabilities.

Most of the time you will need a dynamic architecure but you can start with simple static model and then move to dynamic model training.

eg: 
Spam detection: Dynamic. Static will become stale soon.
Android voice to text : personalize (dynamic) , global (static)
Ad conversion rate: Dynamic.

![image](https://user-images.githubusercontent.com/1594001/138536759-ff9a2093-2dbc-41fa-a524-f850ed500782.png)

Three reference architecures of dynamic training :

#### Using Cloud Function
![image](https://user-images.githubusercontent.com/1594001/138536872-6796efac-f0bb-40b8-bd9e-9d8056482365.png)

#### Using App engine

![image](https://user-images.githubusercontent.com/1594001/138536918-9a86632c-f8ee-48c1-b6c6-c18d5faa9556.png)

#### Dataflow 

![image](https://user-images.githubusercontent.com/1594001/138536969-6d25f408-0de6-4c32-8e7a-ab0551468912.png)


### Serving architecure.
static serving vs dynamic serving.

Minimize average latency. Use static serving (plain lookup table). Use cache ===> Static serving.

Dynamic serving : compute label on demand (CPU intensive) and gives high(variable) latency.

See what is the cost of: (RAM) storage, Latency and CPU.

Know your characterstics of request. namely peakedness and cardinality of the requests.

if cardinality is very low you can use table and use static serving.

![image](https://user-images.githubusercontent.com/1594001/138538594-51e1c24e-e0aa-40e5-96eb-ed375cf8c7f6.png)

eg: 

Spam email detection : high cardinality ==> dynamic serving

android voice to text: dynamic or hybrid

Shopping ad conversation: static. (run batch script in regular interval to update the cache)

Vertex AI platform.
------------
Full support for all all AI task.

* Dataset are stored in cloud storages. (big query or cloud storage)

* Use notebook to develop and deployment your model. (VM is lanuched, GPU / CPU support)

* Use automl or custom training. (Training pipleline)

* Models Create new (autoML or custom) model or import existing model.

* Endpoints, ML models made available for online predictions requests. (Or batch prediction) . REST API exposure.

Lab for end to end implemantion of a ML pipleline in GCP

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/production_ml/babyweight/train_deploy.ipynb


**Module 2**

* Manage data dependencies for ML model. sudden chance in distrubtion of data will change the ML training model entirely.

* Feature should always be scrutinized before added. determine releation of features with label, one feature may be casual and other maybe just correlated with causal one. What happens when correlated feature become decorrelated. Model performance will suffer. Do "leave one feature evaulation" to access impact of feature in label ( This is called ablation analysis).

* changing the distrubiton (label type), also chaning features like adding new zipcode etc in data set will suddenly have lot more unknown zips.

* Make sure you are using your model in interpolation mode not extrapolation mode.
- Be vigilant using monitoring (distrubtion mean/median/variance residual error (prediction-label) is changing as function of input.
- if relation is changing overtime , either retrain model more frequently or use custom loss function which impasize on latest data more. 
- Ablation analysis to reduce the data dependency: Figure out legacy feature (made redundant due to new feature) OR Bundled features (added part of bundle and not valuable individually)
- Carefull about 
  1. data leakage. (data partitioning)
  2. Use of unknown feature at prediction time, while it was present at training time , 
  3. Predict a maj. class

- Have a facility to Rollback model state before the data corruption. This will help you in case when your ML model is trained on corrupt data (transation system going down might give signal no user is buying anything and may skew your model training)
- Model getting stale (new user/new product/ new patterns), espically true for product recommendation system.
- Dynamic retain model in new data

**concept drift** 

data is IID distribution
D is stationary distribtuion
- In real word the distribution of data (features) changes
- concept drift P(Y|X) changes over time. Change in relation of input feature and label.
- prediction drift
- label drift

*Tensorflow data validation tools* (tfdv)
Statistic gen components to visualize the statistics.
schemagen: infer data type.
These two fed to the example gen to validate any drift happening.

![image](https://user-images.githubusercontent.com/1594001/146031034-6dda35cd-d7fe-46aa-be33-552b31b3926d.png)

Use the following notebook to revise the tools for data validation in tensorflow

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/production_ml/solutions/tfdv_basic_spending.ipynb

**Advanced visualization and tensorflow data validation.**

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/production_ml/solutions/tfdv_advanced_taxi.ipynb

uses apache beam to compute stats in parallel.

Visualize data
-----
import tensorflow_data_validation as tfdv
Let say you have train,eval,serving dataset. How do you detect the anamoly in these.
train_stats = tfdv.generate_statistics_from_csv(data_location=TRAIN_DATA)
tfdv.visualize_statistics(train_stats)

Infer schema
------
# Infers schema from the input statistics.
schema = tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema=schema)

Later this schema is used to point anamolies in the data.

You should compare your train/eval/prect data using the diff methods of tfdv
# Compute stats for evaluation data
eval_stats = tfdv.generate_statistics_from_csv(data_location=EVAL_DATA)

# Compare evaluation data with training data
tfdv.visualize_statistics(lhs_statistics=eval_stats, rhs_statistics=train_stats,
                          lhs_name='EVAL_DATASET', rhs_name='TRAIN_DATASET')

**Check for anamolies.**
Given the schema from prior steps. check for anamolies.

anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=schema)
tfdv.display_anomalies(anomalies)
**Fix anamolies**

Use domain knowledge to either add features values to schema or make exception

# Relax the minimum fraction of values that must come from the domain for feature company.
company = tfdv.get_feature(schema, 'company')
company.distribution_constraints.min_domain_mass = 0.9

# Add new value to the domain of feature payment_type.
payment_type_domain = tfdv.get_domain(schema, 'payment_type')
payment_type_domain.value.append('Prcard')

# Validate eval stats after updating the schema 
updated_anomalies = tfdv.validate_statistics(eval_stats, schema)
tfdv.display_anomalies(updated_anomalies)

Detect drift and skew (schema,feature (mainly incorrect encoding), distribution skew)

# Add skew comparator for 'payment_type' feature.
payment_type = tfdv.get_feature(schema, 'payment_type')
payment_type.skew_comparator.infinity_norm.threshold = 0.01

# Add drift comparator for 'company' feature.
company=tfdv.get_feature(schema, 'company')
company.drift_comparator.infinity_norm.threshold = 0.001

# TODO
skew_anomalies = tfdv.validate_statistics(train_stats, schema,
                                          previous_statistics=eval_stats,
                                          serving_statistics=serving_stats)

tfdv.display_anomalies(skew_anomalies)