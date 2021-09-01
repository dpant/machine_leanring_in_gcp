General ML pipelines

- Structure data (table)
    - Explore data: Find out which field (columns) have impact on target variable(mean). So if you have X1,X2,....Xn --> Y figure out which of X1,X2...Xn different value have different mean in Y. If sex(male/female) does not change the mean of Y , it is highly likely its not a good predictor (feature) for Y. You can also figure out how many samples you have for each column value of (X1,X2....Xn) This will help you in figuring out how bananced your dataset is.
    - GCP your data can be usually in bigquery (data warehouse) or GCP storage https://www.coursera.org/learn/end-to-end-ml-tensorflow-gcp/supplement/875zO/ai-platform-notebooks       - How to create a Notebook? In GCP console (console.cloud.google.com) GOTO "AI Platform" ====> Notebooks ==> Enable Notebook API ==> Create Instance. This will take 2-3 mins to launch a notebook session.(Create a VM for it) ==> Open Jupyter Lab.
    - You can create a terminal and clone repo etc in it.

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

