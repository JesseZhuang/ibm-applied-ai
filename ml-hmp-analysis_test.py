
# coding: utf-8

# In[1]:


#!rm -Rf HMP_Dataset
#!git clone https://github.com/wchill/HMP_Dataset


# In[1]:


get_ipython().system(u'ls HMP_Dataset/Brush_teeth')


# In[2]:


from pyspark.sql.types import StructType, StructField, IntegerType

schema = StructType([
    StructField("x", IntegerType(), True),
    StructField("y", IntegerType(), True),
    StructField("z", IntegerType(), True)])

#df = spark.read.csv("user_click_seq.csv",header=False,schema=schema)


# In[3]:


import os

#get list of folders/files in folder HMP_Dataset
file_list = os.listdir('HMP_Dataset')

#filter list for folders containing data
file_list_filtered = [s for s in file_list if '_' in s]
file_list_filtered


# In[4]:


import os

#get list of folders/files in folder HMP_Dataset
file_list = os.listdir('HMP_Dataset')

#filter list for folders containing data
file_list_filtered = [s for s in file_list if '_' in s]

from pyspark.sql.functions import lit

#create pandas data frame for all the data

df = None

for category in file_list_filtered:
    data_files = os.listdir('HMP_Dataset/'+category)
    
    #create a temporary pandas data frame for each data file
    for data_file in data_files:
        print(data_file)
        temp_df = spark.read.option("header", "false").option("delimiter", " ").csv('HMP_Dataset/'+category+'/'+data_file,schema=schema)
        
        #create a column called "source" storing the current CSV file
        temp_df = temp_df.withColumn("source", lit(data_file))
        
        #create a column called "class" storing the current data folder
        temp_df = temp_df.withColumn("class", lit(category))
        
        #append to existing data frame list
        #data_frames = data_frames + [temp_df]
                                                                                                             
        if df is None:
            df = temp_df
        else:
            df = df.union(temp_df)
        


# In[5]:


df.show()


# In[6]:


df.printSchema()


# In[7]:


from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="class",outputCol="classIndex")
indexed = indexer.fit(df).transform(df)
indexed.show()


# In[8]:


from pyspark.ml.feature import OneHotEncoder

encoder = OneHotEncoder(inputCol="classIndex",outputCol="categoryVec")
# categoryVec: sparse vector (12,[3],[1.0]) 12 elements, position 3 there is a 1
encoded = encoder.transform(indexed)
encoded.show()


# In[9]:


from pyspark.ml.feature import VectorAssembler

vectorAssembler = VectorAssembler(inputCols=["x","y","z"],
                                  outputCol="features")
features_vectorized = vectorAssembler.transform(encoded)

features_vectorized.show()


# In[11]:


from pyspark.ml.feature import Normalizer

normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)
normalized_data = normalizer.transform(features_vectorized)
normalized_data.show()


# In[ ]:



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

df = df.repartition(1)
# df.write.parquet(cos.url('hmp.parquet', 'courseraml-donotdelete-pr-qve0ttzezdeodc')) # canonly be written once



# In[17]:


from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[indexer,encoder,vectorAssembler,normalizer])
model = pipeline.fit(df)
prediction = model.transform(df)

prediction.show()


# In[ ]:


df_train = prediction.drop('x').drop('y').drop('z').drop('class').drop('source').drop('features')
df_train.show()

