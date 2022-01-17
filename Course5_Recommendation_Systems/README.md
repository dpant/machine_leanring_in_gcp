# Recommendation systems

**Content based:** you define tags and attach tags with all your contents/items. Later you match user to the most frequently used item in "tag/category"  which he likes. You need metadata(tags etc) about your item. The main task here is to assign proper tags to the content and users.

**Colloborative filtering***
you learn about item and user similarlity with the ratings data (watched,commented,star ratings).
This matix is very large and sparse.idea is to factorize it to store the approximate factors (for memory effeciency). User factor , item factor

![image](https://user-images.githubusercontent.com/1594001/149675599-31b02b0e-34bb-4d67-9acb-71a2d11fd766.png)

**hybrid model**
content based _ collaborative filtering
Most real word system are hybrid
![image](https://user-images.githubusercontent.com/1594001/149675657-edce4c2f-b809-496a-a02b-043df0c71100.png)

You must consider how you train these systems when the ratings/history data comes in continuously.

# Overview.

Identify things which use might like. might recommend things which user have not even searched for ever. 
Take into account location,history,account preference. 
Content discovery is a big thing.

* User : helps us to discover item space. Improve decision space.
* Content provider: Increase user engagement. Change user behaviour

![image](https://user-images.githubusercontent.com/1594001/149676003-fdb8193b-7e16-40d0-912a-ce6d23f8ceb9.png)

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

Content base filtering:
handengineering features for item and learnign how single user align

![image](https://user-images.githubusercontent.com/1594001/149676085-045214ab-3012-4c7b-a568-40f451cfc5b3.png)

collaborative filtering
dont have to rely on hand engineering 

![image](https://user-images.githubusercontent.com/1594001/149676161-cfd5a35b-deba-46de-b3aa-059dc5395790.png)

**knowledge-based:**
When content based or collaborative filtering can't be applied due to insufficient data.
When we don't enough interaction data to know the user enough. Say suggesting user what vaction house to buy next. people don't buy many house in general to hard to extract the preference with user activity.
Knowledge base system will ask user for their preferences and use that information to make recommendation.

Hybrid:

![image](https://user-images.githubusercontent.com/1594001/149676224-ed36d503-cd5e-436d-912c-761cfa2f3cf4.png)

the hybrid model will use a NN to combine all the 3 models.

Neural network based recommendation system.

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

