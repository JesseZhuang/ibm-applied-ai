
# coding: utf-8

# This is the second assignment for the Coursera course "Advanced Machine Learning and Signal Processing"
# 
# Again, please insert to code to your ApacheCouchDB based Cloudant instance below using the "Insert Code" function of Watson Studio( you've done this in Assignment 1 before)
# 
# Done, just execute all cells one after the other and you are done - just note that in the last one you have to update your email address (the one you've used for coursera) and obtain a submission token, you get this from the programming assignment directly on coursera.
# 
# Please fill in the sections labelled with "###YOUR_CODE_GOES_HERE###"

# In[28]:


#your cloudant credentials go here
# @hidden_cell
credentials_1 = {

}


# Let's create a SparkSession object and put the Cloudant credentials into it

# In[29]:


spark = SparkSession    .builder    .appName("Cloudant Spark SQL Example in Python using temp tables")    .config("cloudant.host",credentials_1['custom_url'].split('@')[1])    .config("cloudant.username", credentials_1['username'])    .config("cloudant.password",credentials_1['password'])    .config("jsonstore.rdd.partitions", 1)    .getOrCreate()


# Now it’s time to have a look at the recorded sensor data. You should see data similar to the one exemplified below….
# 

# In[30]:


df=spark.read.load('shake_classification', "com.cloudant.spark")

df.createOrReplaceTempView("df")
spark.sql("SELECT * from df").show()


# Please create a VectorAssembler which consumed columns X, Y and Z and produces a column “features”
# 

# In[31]:


from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Normalizer

indexer = StringIndexer(inputCol="CLASS", outputCol="label")
encoder = OneHotEncoder(inputCol="label", outputCol="labelVec")
vectorAssembler = VectorAssembler(inputCols=["X","Y","Z"],
                                  outputCol="features")
normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)


# Please insatiate a classifier from the SparkML package and assign it to the classifier variable. Make sure to either
# 1.	Rename the “CLASS” column to “label” or
# 2.	Specify the label-column correctly to be “CLASS”
# 

# In[55]:


# LogisticReg accuracy was 54.7% Naive Bayes requires non-negative data
# from pyspark.ml.classification import LogisticRegression
# classifier = LogisticRegression(maxIter=200, regParam=0.2, elasticNetParam=0.8)

from pyspark.ml.classification import GBTClassifier
classifier = GBTClassifier(labelCol="label", featuresCol="features_norm", maxIter=10)


# Let’s train and evaluate…
# 

# In[56]:


from pyspark.ml import Pipeline
# pipeline = Pipeline(stages=[vectorAssembler, classifier])
pipeline = Pipeline(stages=[indexer, encoder, vectorAssembler, normalizer,classifier])


# In[57]:


model = pipeline.fit(df)


# In[58]:


prediction = model.transform(df)


# In[59]:


prediction.show()


# In[60]:


prediction.printSchema()


# In[61]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
binEval = MulticlassClassificationEvaluator().setMetricName("accuracy") .setPredictionCol("prediction").setLabelCol("label")
    
binEval.evaluate(prediction) 


# If you are happy with the result (I’m happy with > 0.55) please submit your solution to the grader by executing the following cells, please don’t forget to obtain an assignment submission token (secret) from the Courera’s graders web page and paste it to the “secret” variable below, including your email address you’ve used for Coursera. (0.55 means that you are performing better than random guesses)
# 

# In[62]:


get_ipython().system(u'rm -Rf a2_m2.json')


# In[63]:


prediction = prediction.repartition(1)
prediction.write.json('a2_m2.json')


# In[64]:


get_ipython().system(u'rm -f rklib.py')
get_ipython().system(u'wget https://raw.githubusercontent.com/romeokienzler/developerWorks/master/coursera/ai/rklib.py')


# In[65]:


get_ipython().system(u'zip -r a2_m2.json.zip a2_m2.json')


# In[66]:


get_ipython().system(u'base64 a2_m2.json.zip > a2_m2.json.zip.base64')


# In[67]:


from rklib import submit
key = "J3sDL2J8EeiaXhILFWw2-g"
part = "G4P6f"
email = ""
secret = ""

with open('a2_m2.json.zip.base64', 'r') as myfile:
    data=myfile.read()
submit(email, secret, key, part, [part], data)

