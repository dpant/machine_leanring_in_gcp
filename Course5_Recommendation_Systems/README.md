# Recommendation systems

**Content based:** you define tags and attach tags with all your contents/items. Later you match user to the most frequently used item in "tag/category"  which he liked (in past). Notice we don't rely on other users interactions to recommend things to the user. It's just user interactions. You will need metadata(tags etc) about your item. The main task here is to assign proper tags to the content and users and match them. Also most of your data/metadata/tags on items is curated by experts not by users interaction with the product (This is the major difference from colloborative filtering)

**Colloborative filtering**
you learn about item and user similarlity with the ratings data (watched,commented,star ratings).
This matix is very large and sparse.idea is to factorize it to store the approximate factors (for memory effeciency). User factor , item factor

![image](https://user-images.githubusercontent.com/1594001/149675599-31b02b0e-34bb-4d67-9acb-71a2d11fd766.png)

**Hybrid model**
content based + collaborative filtering + knowledge based + .....
Most real word system are hybrid
![image](https://user-images.githubusercontent.com/1594001/149675657-edce4c2f-b809-496a-a02b-043df0c71100.png)

You must consider how you train these systems when the ratings/history data comes in continuously.

# Content based recommendation systems (Overview).

Identify things which user might like. Also should be able to recommend things which user have not even searched for. 
Take into account location,history,account preference. 
Content discovery is a big thing! (colloborative filtering does this very nicely)

* User : helps us to discover item space. Improve decision space.
* Content provider: Increase user engagement. Change user behaviour

![image](https://user-images.githubusercontent.com/1594001/149676003-fdb8193b-7e16-40d0-912a-ce6d23f8ceb9.png)
```
Features:
properties of users (m/f)
properties of house 
Previous rentals of user
previous rentals of house
compare similar users
compare similar items.
Targets:
rating for unseen properties.
next rental property.
```

Content base filtering:
hand-engineering features for item and learnign how single user align

![image](https://user-images.githubusercontent.com/1594001/149676085-045214ab-3012-4c7b-a568-40f451cfc5b3.png)

collaborative filtering: Dont have to rely on hand engineering item features (tags)

![image](https://user-images.githubusercontent.com/1594001/149676161-cfd5a35b-deba-46de-b3aa-059dc5395790.png)

**knowledge-based:**
When content based or collaborative filtering can't be applied due to insufficient data.
When we don't enough interaction data to know the user enough. Say suggesting user what vaction house to buy next. people don't buy many house in general to hard to extract the preference with user activity.
Knowledge base system will ask user for their preferences and use that information to make recommendation.

Hybrid:

![image](https://user-images.githubusercontent.com/1594001/149676224-ed36d503-cd5e-436d-912c-761cfa2f3cf4.png)

the hybrid model will use a NN to combine all the 3 models.

## Neural network based recommendation system.


![image](https://user-images.githubusercontent.com/1594001/149829785-5b65294e-cb31-450b-b7cb-d048cf9df2f4.png)


![image](https://user-images.githubusercontent.com/1594001/149676345-492677cb-7947-488d-85c4-4052b826b097.png)

**pitfalls** 

![image](https://user-images.githubusercontent.com/1594001/149676467-d447c1f9-2c8b-4fcc-85ad-f9720ab6327d.png)
![image](https://user-images.githubusercontent.com/1594001/149676494-ced575f4-a343-4ce6-b53b-a36cf8dbba15.png)


cold start problem  (Go for content based in this case)
![image](https://user-images.githubusercontent.com/1594001/149676525-3d7a9d3d-c427-4fef-971a-970bc3421379.png)

explicit feedback is not available for majority of activity.

![image](https://user-images.githubusercontent.com/1594001/149676553-f952e941-42b0-4bc4-87a8-32f1c7613f84.png)

generate a model (NN) to predict the explicity rating and feed it to the recommendation system.

![image](https://user-images.githubusercontent.com/1594001/149676580-2ed549d3-f6df-4c67-b0e8-104e654571ca.png)


# Content based recommendation systems:

Use item features (tag) + user past likes to recommend new items. does not take into account other users preferences/interactions.
Check what movies does the user rate highly and recommend one which are more similar to the highly rated movies/item.

Cold start? if user does not have past activity hard to get the user factor.

How to do you measure similarity?

Consider different themes / geners and see the overlap
if two user like same movies they are similar.

Get embedding of user/item and get the vector representation for user (This is entirely based on past interation of the user with the item, hence effected by cold start). Use dot product or cosine similarity between them movies(items).

![image](https://user-images.githubusercontent.com/1594001/149824856-3d05f32c-0668-4cfd-8673-fe2e28b325b8.png)


![image](https://user-images.githubusercontent.com/1594001/149824875-e023a5a3-0b20-4bc2-9086-65f8cc89cd1d.png)

Building user vector
---
How do you find the embedding of User rating and item features given.

![image](https://user-images.githubusercontent.com/1594001/149825169-064294e0-54b5-4f78-bc65-499e165ef468.png)

Scale the movie feature matrix by user ratings.sum acorss the feature column and normalize

![image](https://user-images.githubusercontent.com/1594001/149825282-e0295df0-01eb-4ac0-b2ea-9214d81be069.png)

![image](https://user-images.githubusercontent.com/1594001/149825338-3930d045-7c15-4e77-87f9-7c0d8891b53a.png)

# Making a recommendation
use dot product to measure the similarliy of unrated movies. multiple the user vector to the movie feature matix

![image](https://user-images.githubusercontent.com/1594001/149827740-ca9fa0f7-8835-41ad-9628-9608d70ab672.png)

Make the value by sorting to make recommendation.

# How to scale this for multiple user at a time?

![image](https://user-images.githubusercontent.com/1594001/149827970-ca747135-4d37-4541-b61d-24c72b8d1596.png)

![image](https://user-images.githubusercontent.com/1594001/149828015-62aac2d8-c7c6-4c42-9c6a-0b4f4f903a19.png)

Get weighted user matrix for all users. stack them

![image](https://user-images.githubusercontent.com/1594001/149828103-96094258-ea23-46eb-9a27-7a7714029f52.png)

![image](https://user-images.githubusercontent.com/1594001/149828207-fc17ce2b-fbce-4ed0-b89b-e3cd644cd383.png)


![image](https://user-images.githubusercontent.com/1594001/149828559-0cbe8d7c-7107-4d0c-9e07-fc1e4f0504b8.png)


difficult to expand intreast of user using content based recommendation systems. If use have not rated some of the features (say generas) of movies it will never show up in the list. 

lab: content based recommendation system using low level tf functions.
https://www.coursera.org/learn/recommendation-models-gcp/lecture/NfhNj/lab-solution-create-a-content-based-recommendation-system

Lab git link:
https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/10_recommend/content_based_by_hand.ipynb



# Using neural network to get conent based recommendation system.

Consider user feature and movie features.

![image](https://user-images.githubusercontent.com/1594001/149829785-5b65294e-cb31-450b-b7cb-d048cf9df2f4.png)

kurier.at recommendation system.

![image](https://user-images.githubusercontent.com/1594001/150024981-60b27ccf-ea66-471f-b493-937db91402cc.png)

visitor_id: depend of your browser id etc. Not used in model.
content_id: article id. categorical column. with hash bucket. categorical_column_with_hash_bucket()
category column: categorical_column_with_vacabulary_list()
title: use tf hub to embed current article
author: categorical_column_with_hash_bucket
months_since_epoch: numeric_column()

![image](https://user-images.githubusercontent.com/1594001/150027323-d4465489-2c1b-4fda-9ebb-8539dc122c7a.png)

Walkthrough of lab:

https://www.coursera.org/learn/recommendation-models-gcp/lecture/tWx6t/lab-solution-create-a-content-based-recommendation-system-using-a-neural-network

reaches 30% of accuracy after training.


# Collaborative filtering

Use WALS (weighted average least square)

Shortcoming of content based recommendation system: "Search local neighbourhood" stays in local neighbourhood. what if you don't have user interest.

One user manifold might be small so might not have much to recommend. How about learning from other users. (this is done by collaborative filtering)

Explicit feedback/ratings: (user interaction matrix)

![image](https://user-images.githubusercontent.com/1594001/150899705-c10b4c82-e57a-4214-8e2f-c307c7f2e5fa.png)

![image](https://user-images.githubusercontent.com/1594001/150899850-9ba9da57-e466-4933-933f-dff327cf2e82.png)

We can get also implicity feedback.

# embedding users and items in same space.

Users and items are represented in d-dimentional embedding space. (Same embeddign space)

![image](https://user-images.githubusercontent.com/1594001/150900409-bad467c4-1146-478e-8511-c5e5fa6b4577.png)

How do you get the user factor embedding and item embedding.==> Use the user-item interaction matrix data.


How to recommend?

find closest movie which is not watched by taking dot product.

![image](https://user-images.githubusercontent.com/1594001/150900655-68178c15-470b-4e14-a34d-9f53eda35799.png)

Matrix factorization is key here as the interaction matrix is very sparse.

10K movies 50 million users. 500 Billion interaction (cells in matrix). Factorization will help. If k latent feature ==> k (user + movies) {total cells}

![image](https://user-images.githubusercontent.com/1594001/150901165-761d3063-69dd-422f-9337-34c026dabafc.png)

Factorization approaches.

![image](https://user-images.githubusercontent.com/1594001/150901487-58ef48df-30eb-48e2-aa5a-5fc343f532e9.png)

Approximate way to find matrix factoring.

![image](https://user-images.githubusercontent.com/1594001/150901571-61a1f64c-4f08-4b2c-9779-d4b9bc88fca7.png)

How to find the minimization (squared error). "Least square error minimization problem"

Use SGD? (Generalist algo to solve the minimization problems)
Flexible,parallel (how), many iterations to converge (slow). SGD can't handle missing interaction pair of user-item.

Use Alternativing least square (ALS)?
parallalizable, can be used for only on "least square problem" (which fits our matrix factorization case), easy to handle un-observed / missign interaction pair.

How to treat missing interaction data.
SVD. unobserved pair are assigned 0 value. This is not same as missing value but like SGD, SVD can't handle missing data

![image](https://user-images.githubusercontent.com/1594001/150902293-3b959d1a-7d5a-4459-804d-06138e5800ee.png)

ALS:
Can Ignore the missings values

WALS: Use weights instead of 0.

![image](https://user-images.githubusercontent.com/1594001/150902378-a16246ba-daf9-4f6e-bcb0-9e072faf7fae.png)

ALS algorithm: (learn the details)

![image](https://user-images.githubusercontent.com/1594001/150903036-3f434f00-c77e-401c-809c-9e81820b1c58.png)

Notice the alternating loop inside. fix u find v. fix v find u.

## How to prepare the input data for ALS factorization
Data is usually represented as table not matrix ( as matrix is too sparse)

WALS requires the ratings matrix to be really matrix entries so you will have to map id strings to be 0,1,2....

![image](https://user-images.githubusercontent.com/1594001/150918443-ad85692e-c8d2-4aed-b895-6118de03666d.png)

![image](https://user-images.githubusercontent.com/1594001/150918394-73c562ce-fe0d-4bfb-bc70-b98777af6711.png)

session_duration is implicit feedback. (longer session mean user like the videos). Notice the visitor id and content id are just large string not continous index of matrix.

Also you need to have access to user input dataset (as you want to filterout existing watch history etc)

![image](https://user-images.githubusercontent.com/1594001/150918666-2e919d70-6c5b-4b96-a569-202ad7854dae.png)

# implementing ALS in tf

preprocessing data. store sparse tensors (stored as index and values) for rows and columns.
so key is rowid (say user), indices is kind of (column id) which is itemids and values is the rating.

![image](https://user-images.githubusercontent.com/1594001/150919605-a2778798-e06d-4b96-9e8a-d505cb21579e.png)

![image](https://user-images.githubusercontent.com/1594001/150920407-b2991c6c-3971-454c-888f-7428321455e1.png)



Dealing with cold start problem
=====
Hard to deal with cold start(new user or new item added) problem in collaborative filtering.

Use hybrid model (or content based/knowledge based) or a very simple flow chart like below.

![image](https://user-images.githubusercontent.com/1594001/150922629-bc25df19-1f74-4da6-bc16-dd5b17abcea1.png)

https://www.coursera.org/learn/recommendation-models-gcp/lecture/KmRBf/cold-starts


Hybrid recommendation system.
-----------

Notice if you just have user interaction data than collaborative filtering.
if the data(tags) is generated by "product experts curators" then content based.
explicity asking user for preference ==> knowledge based.

![image](https://user-images.githubusercontent.com/1594001/150924419-c0318205-bc08-4d71-9049-843aee64243b.png)

Combine all the models.

![image](https://user-images.githubusercontent.com/1594001/151038422-252ba6fb-1949-4d17-8917-a488290cf202.png)

Hybrid recommendation system for movie recommendation.
------

## Content based system data:

https://www.coursera.org/learn/recommendation-models-gcp/lecture/Ik2VK/lab-designing-a-hybrid-recommendation-system

structured
Generes (N hot encoding)
Themes.  (N hot encoding)
Actor director involved. (N hot encoding)
professional ratings (notice professional like imdb)

Unstructured 
movie summary text (Do NLP think car chase in summary)
Stills from movie. (use CNN to process and generate tags)
moview trailers
Professional review (not writtend by users)

## content based

https://www.coursera.org/learn/recommendation-models-gcp/lecture/Nb79i/lab-designing-a-hybrid-collaborative-filtering-recommendation-system

Structured: 

User rating (Notice other user) explicit rating.
Movie watch time
Skipped amount
User view 
user wishlist
cart history
purchase history/return history

Unstructured data:
User reviews (sentiment analysis run on this )
User answers questions (sentiment analysis run on this )

# knowledge based.

https://www.coursera.org/learn/recommendation-models-gcp/lecture/NNYLb/lab-designing-a-hybrid-knowledge-based-recommendation-system


Structred:
Demographic information (age , sex etc)
User location, country , laguage etc
Genere preferences
Global filters ( give move which are higher then 4 rated etc)

Unstructured:
User about me (do NLP and find interests)

![image](https://user-images.githubusercontent.com/1594001/151041805-ca459859-da16-4103-9607-3beb146ece77.png)

![image](https://user-images.githubusercontent.com/1594001/151042253-a58a6e3f-d082-4f9d-be65-8b83d9b83c59.png)

Lab: bigquery ml recommendation system using ALS

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/recommendation_systems/solutions/als_bqml_hybrid.ipynb


# Context aware recommendations systems (CARS)

Recommendation very much dependent on context for example for a movie recommendation
```
mood at the time?
special occasion?
where are you watching it (home/theater)
Same place different time differnt rating ==> different recommendation.
```
(User X items) --> ratings ( tradiational collaborative filtering recommendation systems)

(Users x items x contexts) --> Ratings (CARS)

![image](https://user-images.githubusercontent.com/1594001/151269842-3b752b6e-c90f-442b-a1c1-ef40653def6b.png)

![image](https://user-images.githubusercontent.com/1594001/151269952-927476b3-4b6f-45dd-ba36-d02602cceffa.png)

# CARS (Pre-filtering to handle the contexual data)

Contexual data(UxIxC) ------> Prefiltering -----> transformed data (UxI) ----> "Traditional recommendation systems"

![image](https://user-images.githubusercontent.com/1594001/151270868-d5c84081-646b-498f-834b-ddf4d6c104b8.png)

Key is how to split the data (item splitting,user splitting and item-user splitting)?

![image](https://user-images.githubusercontent.com/1594001/151271002-6d8d1a16-6e93-4f57-8fae-e89b8b7a77fd.png)

Look at rating weekend ratings are very high compare to weekday ratings.(same users, same items)

![image](https://user-images.githubusercontent.com/1594001/151271090-640d5c0b-2459-4443-812b-aea20eee397b.png)

It will aprear as item is merged with context(when) to create multple split item data. its item-context pair.

Use t test to determine what split maximize the t value.

You can also do user splitting and it will apear as separate user interacting with item. its user-context pair.

you can do both the split together
"user-time(context)" and item-location(context)

![image](https://user-images.githubusercontent.com/1594001/151271552-7714aae4-19f0-423f-8fff-887870063ac1.png)

## Contextual postfiltering

Ignore the context during the recommendation phase.

![image](https://user-images.githubusercontent.com/1594001/151271869-9848bfaf-8e4e-43be-84fc-0d8d2a6b9ee6.png)

later do adjustment on the recommendation ranking by re-introducing context. Filter things which is not in current context. OR alterning ranking. Use heuristic  or another model for this.

Contextual prob is found for user i , item j in context c = Multiply the non-contexualized rating with a probablity to get the contexualize rating.

# Context aware recommendations

Handle context in the recommendation systems algorithm itself. 

![image](https://user-images.githubusercontent.com/1594001/153130127-f149052b-94a9-43c7-8cbd-8b866dffc390.png)

different contexts:

![image](https://user-images.githubusercontent.com/1594001/153130305-048138b5-0130-4409-9ded-15a9aadfcaa5.png)

Calculate context rating deviation (CRD)

![image](https://user-images.githubusercontent.com/1594001/153130329-20640904-83c7-4230-9f48-5ae9bf19d0ad.png)

recommenation system without context.

![image](https://user-images.githubusercontent.com/1594001/153130687-b200ebfc-e9a4-42c4-8ad3-22a84b575aa0.png)

adding CRD term
![image](https://user-images.githubusercontent.com/1594001/153130885-bf06edaa-0ef6-434d-a44b-c1e868998a8b.png)


# Youtube recommendations

![image](https://user-images.githubusercontent.com/1594001/153130996-41c83a31-0195-4188-8d0e-8bdd45254848.png)

Candidate generation NN : tuned for high precision
Ranking NN : tuned for higher recall.

![image](https://user-images.githubusercontent.com/1594001/153131198-330cc04b-9b6d-44bb-8410-ad94f0dc2a9b.png)

![image](https://user-images.githubusercontent.com/1594001/153131294-43c4b265-e6fb-4153-8c21-231a5441e722.png)

![image](https://user-images.githubusercontent.com/1594001/153131438-46488728-c300-4fc5-b6d5-71d7fdc051eb.png)

Ranking NN:

![image](https://user-images.githubusercontent.com/1594001/153131926-ba180f05-a490-40f1-a2c8-0550e6d1b715.png)

![image](https://user-images.githubusercontent.com/1594001/153131719-46dcb361-be64-490f-9313-3d4a4703ae6d.png)

![image](https://user-images.githubusercontent.com/1594001/153131763-0d2de54a-559a-4d9b-a5c8-4b6b1b8da338.png)

unit weight for negative example and watch time for the postive example.

![image](https://user-images.githubusercontent.com/1594001/153132076-b68f431e-6f9c-495c-b18e-8cff4251cd92.png)

# Deploying recommendation system in cloud.

How to build pipleline for continuous train (re-train) and deployment.

How do you trust your upstream pipeline which is providing data. How do you put measure that bugs in upstream does not effect your model training. Health check building.

Use google cloud composer (Apache airflow) for orchestration

High level architechure of deployment:

Data (Google analytics) Big Query ---> Exported to Google cloud storage (GCS as csv) --> Cloud ML engine -----> training model ----> deploy in app engine.

![image](https://user-images.githubusercontent.com/1594001/153245520-a65814b1-31c9-4f87-ad9e-8c9a5ed82cb5.png)

For recommendation systems:

Use batch prediction (end of day or every hour. One recommendation is hard to scale and unnecessary) 
How to autmatically refresh our training data and ML model? Use cloud composer (Apache Airflow). You can do perodic trigger or event based (when data comes in)
Cloud function watch ---> trigger cloud compoer workflow to put the data in big query.

Cloud composer:
Model your workflow as a DAG (dependency) and airflow will execute it. Built in retry, healthcheck , email notificatnoi etc makes it very useful tool.
There is a simple airflow webserver UI which you can see to manage and look at your pipeline (execution) in GCP.

![image](https://user-images.githubusercontent.com/1594001/153247776-ac5059ef-0e57-458d-a1f4-02f5ae0d056a.png)

https://www.coursera.org/learn/recommendation-models-gcp/lecture/tY8Qb/cloud-composer-overview

Airflow: main DAG is specified as python file with operators.

![image](https://user-images.githubusercontent.com/1594001/153252064-e536aee7-6645-48de-aff4-c0be98018e9b.png)

Use healthcheck operator aka BigQueryCheckOperator().... If fail , it halts pipeline and send (email) notification.
What if data is itself corrupt (non zero data but corrupt) put interval check (bands of values) in bigquery to prevent corrupt data making your ML model corrupt.

The Cloud composer flow can be either trigrred based on time interval (pull based) or event driven (push based using cloud function)

https://www.coursera.org/learn/recommendation-models-gcp/lecture/9OxzE/cloud-composer-triggering-workflows-with-cloud-functions

cloud flow
https://www.coursera.org/learn/recommendation-models-gcp/lecture/9OxzE/cloud-composer-triggering-workflows-with-cloud-functions


Debugging cloud function and airflow

https://www.coursera.org/learn/recommendation-models-gcp/lecture/AlIJ7/cloud-composer-monitoring-and-logging

matrix factarization and recommendation system, including store data in sparsetensor etc.
https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/10_recommend/cf_softmax_model/solution/cfmodel_softmax_model_solution.ipynb


Notebook about advance recommendation systems:
https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive2/recommendation_systems/solutions


Nice summary of full specialization 
-----------

https://www.coursera.org/learn/recommendation-models-gcp/lecture/4TN6G/specialization-summary

