
# coding: utf-8

# This is the last assignment for the Coursera course "Advanced Machine Learning and Signal Processing"
# 
# Again, please insert to code to your ApacheCouchDB based Cloudant instance below using the "Insert Code" function of Watson Studio (you've done this in Assignment 1 before)
# 
# 
# Done, just execute all cells one after the other and you are done - just note that in the last one you should update your email address (the one you've used for coursera) and obtain a submission token, you get this from the programming assignment directly on coursera.
# 
# Please fill in the sections labelled with "###YOUR_CODE_GOES_HERE###"
# 
# The purpose of this assignment is to learn how feature engineering boosts model performance. You will apply Discrete Fourier Transformation on the accelerometer sensor time series and therefore transforming the dataset from the time to the frequency domain. 
# 
# After that, you’ll use a classification algorithm of your choice to create a model and submit the new predictions to the grader. Done.
# 

# In[9]:


#your cloudant credentials go here
# @hidden_cell
credentials_1 = {
  'password':"""be323b12c38d480b142c2005c2975c2745d4bb75d68825d086b262c57cc4ce86""",
  'custom_url':'https://81f7f907-00a0-455a-9b93-8ed9fa0b118c-bluemix:be323b12c38d480b142c2005c2975c2745d4bb75d68825d086b262c57cc4ce86@81f7f907-00a0-455a-9b93-8ed9fa0b118c-bluemix.cloudantnosqldb.appdomain.cloud',
  'username':'81f7f907-00a0-455a-9b93-8ed9fa0b118c-bluemix',
  'url':'https://undefined'
}


# Let's create a SparkSession object and put the Cloudant credentials into it

# In[10]:


spark = SparkSession    .builder    .appName("Cloudant Spark SQL Example in Python using temp tables")    .config("cloudant.host",credentials_1['custom_url'].split('@')[1])    .config("cloudant.username", credentials_1['username'])    .config("cloudant.password",credentials_1['password'])    .config("jsonstore.rdd.partitions", 1)    .getOrCreate()


# Now it’s time to read the sensor data and create a temporary query table.

# In[11]:


df=spark.read.load('shake_classification', "com.cloudant.spark")
df.createOrReplaceTempView("df")
df.show()


# We need to make sure SystemML is installed.
# 

# In[ ]:


get_ipython().system(u'pip install systemml')


# We’ll use Apache SystemML to implement Discrete Fourier Transformation. This way all computation continues to happen on the Apache Spark cluster for advanced scalability and performance.

# In[12]:


from systemml import MLContext, dml
ml = MLContext(spark)


# As you’ve learned from the lecture, implementing Discrete Fourier Transformation in a linear algebra programming language is simple. Apache SystemML DML is such a language and as you can see the implementation is straightforward and doesn’t differ too much from the mathematical definition (Just note that the sum operator has been swapped with a vector dot product using the %*% syntax borrowed from R
# ):
# 
# <img style="float: left;" src="https://wikimedia.org/api/rest_v1/media/math/render/svg/1af0a78dc50bbf118ab6bd4c4dcc3c4ff8502223">
# 
# 

# In[13]:


dml_script = '''
PI = 3.141592654
N = nrow(signal)

n = seq(0, N-1, 1)
k = seq(0, N-1, 1)

M = (n %*% t(k))*(2*PI/N)

Xa = cos(M) %*% signal
Xb = sin(M) %*% signal

DFT = cbind(Xa, Xb)
'''


# Now it’s time to create a function which takes a single row Apache Spark data frame as argument (the one containing the accelerometer measurement time series for one axis) and returns the Fourier transformation of it. In addition, we are adding an index column for later joining all axis together and renaming the columns to appropriate names. The result of this function is an Apache Spark DataFrame containing the Fourier Transformation of its input in two columns. 
# 

# In[14]:


from pyspark.sql.functions import monotonically_increasing_id

def dft_systemml(signal,name):
    prog = dml(dml_script).input('signal', signal).output('DFT')
    
    return (

    #execute the script inside the SystemML engine running on top of Apache Spark
    ml.execute(prog) 
     
         #read result from SystemML execution back as SystemML Matrix
        .get('DFT') 
     
         #convert SystemML Matrix to ApacheSpark DataFrame 
        .toDF() 
     
         #rename default column names
        .selectExpr('C1 as %sa' % (name), 'C2 as %sb' % (name)) 
     
         #add unique ID per row for later joining
        .withColumn("id", monotonically_increasing_id())
    )
        




# In[29]:


df_x = df.filter(df.CLASS == 1).select('X', 'CLASS')
df_x.show()
dft_x1 = dft_systemml(df_x, 'x')
dft_x1.show()

dft_y1 = dft_systemml(df.filter(df.CLASS == 1).select('Y'), 'y')
dft_z1 = dft_systemml(df.filter(df.CLASS == 1).select('Z'), 'z')


# Now it’s time to create DataFrames containing for each accelerometer sensor axis and one for each class. This means you’ll get 6 DataFrames. Please implement this using the relational API of DataFrames or SparkSQL.
# 

# In[31]:


# x0 = dft_systemml(df.filter(df.CLASS == 0).select('X'), 'x')
# y0 = dft_systemml(df.filter(df.CLASS == 0).select('Y'), 'y')
# z0 = dft_systemml(df.filter(df.CLASS == 0).select('Z'), 'z')
# x1 = dft_systemml(df.filter(df.CLASS == 1).select('X'), 'x')
# y1 = dft_systemml(df.filter(df.CLASS == 1).select('Y'), 'y')
# z1 = dft_systemml(df.filter(df.CLASS == 1).select('Z'), 'z')

c0 = df.filter(df.CLASS == 0)
c1 = df.filter(df.CLASS == 1)
x0 = c0.select('X')
y0 = c0.select('Y')
z0 = c0.select('Z')
x1 = c1.select('X')
y1 = c1.select('Y')
z1 = c1.select('Z')


# Since we’ve created this cool DFT function before, we can just call it for each of the 6 DataFrames now. And since the result of this function call is a DataFrame again we can use the pyspark best practice in simply calling methods on it sequentially. So what we are doing is the following:
# 
# - Calling DFT for each class and accelerometer sensor axis.
# - Joining them together on the ID column. 
# - Re-adding a column containing the class index.
# - Stacking both Dataframes for each classes together
# 
# 

# In[32]:


from pyspark.sql.functions import lit

df_class_0 = dft_systemml(x0,'x')     .join(dft_systemml(y0,'y'), on=['id'], how='inner')     .join(dft_systemml(z0,'z'), on=['id'], how='inner')     .withColumn('class', lit(0))
    
df_class_1 = dft_systemml(x1,'x')     .join(dft_systemml(y1,'y'), on=['id'], how='inner')     .join(dft_systemml(z1,'z'), on=['id'], how='inner')     .withColumn('class', lit(1))

df_dft = df_class_0.union(df_class_1)

df_dft.show()


# Please create a VectorAssembler which consumes the newly created DFT columns and produces a column “features”
# 

# In[38]:


from pyspark.ml.feature import VectorAssembler

vectorAssembler = VectorAssembler(inputCols=["xa", "ya", "za", "xb", "yb", "zb"],outputCol="features")


# Please insatiate a classifier from the SparkML package and assign it to the classifier variable. Make sure to set the “class” column as target.
# 

# In[34]:


sc.version


# In[42]:


from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

layers = [6, 10, 10, 2]

classifier = MultilayerPerceptronClassifier(featuresCol='features',labelCol='class',maxIter=100, layers=layers, blockSize=128, seed=1234)


# Let’s train and evaluate…
# 

# In[43]:


from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, classifier])


# In[44]:


model = pipeline.fit(df_dft)


# In[45]:


prediction = model.transform(df_dft)


# In[46]:


prediction.show()


# In[47]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
binEval = MulticlassClassificationEvaluator().setMetricName("accuracy") .setPredictionCol("prediction").setLabelCol("class")
    
binEval.evaluate(prediction) 


# If you are happy with the result (I’m happy with > 0.8) please submit your solution to the grader by executing the following cells, please don’t forget to obtain an assignment submission token (secret) from the Courera’s graders web page and paste it to the “secret” variable below, including your email address you’ve used for Coursera. 
# 

# In[48]:


get_ipython().system(u'rm -Rf a2_m4.json')


# In[49]:


prediction = prediction.repartition(1)
prediction.write.json('a2_m4.json')


# In[50]:


get_ipython().system(u'rm -f rklib.py')
get_ipython().system(u'wget https://raw.githubusercontent.com/romeokienzler/developerWorks/master/coursera/ai/rklib.py')


# In[51]:


get_ipython().system(u'zip -r a2_m4.json.zip a2_m4.json')


# In[52]:


get_ipython().system(u'base64 a2_m4.json.zip > a2_m4.json.zip.base64')


# In[53]:


from rklib import submit
key = "-fBiYHYDEeiR4QqiFhAvkA"
part = "IjtJk"
email = ""
secret = ""

with open('a2_m4.json.zip.base64', 'r') as myfile:
    data=myfile.read()
submit(email, secret, key, part, [part], data)

