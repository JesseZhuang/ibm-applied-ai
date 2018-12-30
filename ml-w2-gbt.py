
# coding: utf-8

# In[2]:


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


# In[3]:


df.createOrReplaceTempView('df')
df_two_class = spark.sql("select * from df where class in ('Use_telephone','Standup_chair')")


# In[4]:


splits = df_two_class.randomSplit([0.8, 0.2])
df_train = splits[0]
df_test = splits[1]


# In[5]:


from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Normalizer


indexer = StringIndexer(inputCol="class", outputCol="label")

vectorAssembler = VectorAssembler(inputCols=["x","y","z"],
                                  outputCol="features")

normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)


 



# In[6]:


from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)


# In[7]:



from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[indexer, vectorAssembler, normalizer,gbt])


# In[8]:


model = pipeline.fit(df_train)


# In[9]:


prediction = model.transform(df_train)


# In[10]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
binEval = MulticlassClassificationEvaluator().setMetricName("accuracy") .setPredictionCol("prediction").setLabelCol("label")
    
binEval.evaluate(prediction) # 91%, worse than SVM


# In[11]:


prediction = model.transform(df_test)


# In[12]:


binEval.evaluate(prediction) # 91%, not overfitting


# In[13]:


from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

paramGrid = ParamGridBuilder()     .addGrid(normalizer.p, [1.0, 2.0, 10.0])     .addGrid(gbt.maxBins, [2,4,8,16])     .addGrid(gbt.maxDepth, [2,4,8,16])     .build()


# In[14]:


crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=4)


# In[15]:


# take ages, exponential
cvModel = crossval.fit(df_train)


# In[16]:


prediction = cvModel.transform(df_test)


# In[17]:


binEval.evaluate(prediction)

