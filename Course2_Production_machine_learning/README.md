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
Minimize average latency.
Use cache ===> Static serving.









