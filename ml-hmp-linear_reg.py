
# coding: utf-8

# In[3]:


import ibmos2spark

# @hidden_cell
credentials = {
    'endpoint': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
    'api_key': 'iTf***',
    'service_id': 'iam-ServiceId-d4b06e46-293a-4417-b76c-2f16076a9353',
    'iam_service_endpoint': 'https://iam.ng.bluemix.net/oidc/token'}

configuration_name = 'os_b0f1407510994fd1b793b85137baafb8_configs'
cos = ibmos2spark.CloudObjectStorage(sc, credentials, configuration_name, 'bluemix_cos')

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
# Since JSON data can be semi-structured and contain additional metadata, it is possible that you might face issues with the DataFrame layout.
# Please read the documentation of 'SparkSession.read()' to learn more about the possibilities to adjust the data loading.
# PySpark documentation: http://spark.apache.org/docs/2.0.2/api/python/pyspark.sql.html#pyspark.sql.DataFrameReader.json

df = spark.read.parquet(cos.url('hmp.parquet', 'courseraml-donotdelete-pr-qve0ttzezdeodc'))
df.show()


# In[5]:


df.createOrReplaceTempView('df')

# linear regresssion
df_energy = spark.sql("""
select sqrt(sum(x*x)+sum(y*y)+sum(z*z)) as label, class from df group by class
""")
df_energy.show()
df_energy.createOrReplaceTempView('df_energy')


# In[6]:


df_join = spark.sql("""
select * from df inner join df_energy on df.class=df_energy.class
""")
df_join.show()


# In[7]:


from pyspark.ml.feature import VectorAssembler, Normalizer

vectorAssembler = VectorAssembler(inputCols=["x","y","z"],
                                  outputCol="features")
normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)


# In[8]:


from pyspark.ml.regression import LinearRegression

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)


# In[9]:


from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, normalizer,lr])


# In[10]:


model = pipeline.fit(df_join)
prediction = model.transform(df_join)

prediction.show()


# In[12]:


model.stages[2].summary.r2

