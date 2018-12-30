
# coding: utf-8

# # Assignment 3
# 
# Welcome to Assignment 3. This will be even more fun. Now we will calculate statistical measures on the test data you have created.
# 
# YOU ARE NOT ALLOWED TO USE ANY OTHER 3RD PARTY LIBRARIES LIKE PANDAS. PLEASE ONLY MODIFY CONTENT INSIDE THE FUNCTION SKELETONS
# Please read why: https://www.coursera.org/learn/exploring-visualizing-iot-data/discussions/weeks/3/threads/skjCbNgeEeapeQ5W6suLkA
# . Just make sure you hit the play button on each cell from top to down. There are seven functions you have to implement. Please also make sure than on each change on a function you hit the play button again on the corresponding cell to make it available to the rest of this notebook.
# Please also make sure to only implement the function bodies and DON'T add any additional code outside functions since this might confuse the autograder.
# 
# So the function below is used to make it easy for you to create a data frame from a cloudant data frame using the so called "DataSource" which is some sort of a plugin which allows ApacheSpark to use different data sources.
# 

# All functions can be implemented using DataFrames, ApacheSparkSQL or RDDs. We are only interested in the result. You are given the reference to the data frame in the "df" parameter and in case you want to use SQL just use the "spark" parameter which is a reference to the global SparkSession object. Finally if you want to use RDDs just use "df.rdd" for obtaining a reference to the underlying RDD object. 
# 
# Let's start with the first function. Please calculate the minimal temperature for the test data set you have created. We've provided a little skeleton for you in case you want to use SQL. You can use this skeleton for all subsequent functions. Everything can be implemented using SQL only if you like.

# In[1]:


def minTemperature(df,spark):
    # return spark.sql("SELECT ##INSERT YOUR CODE HERE## as mintemp from washing").first().mintemp
    # return df.rdd.map(lambda d: d['temperature']).min() # Py4JJavaError None cannot be compard to int
    # return df.rdd.min(key='temperature')
    return df.rdd.map(lambda d: d['temperature']).filter(lambda x: x != None).min()


# Please now do the same for the mean of the temperature

# In[2]:


def meanTemperature(df,spark):
    return df.rdd.map(lambda d: d['temperature']).filter(lambda x: x != None).mean()


# Please now do the same for the maximum of the temperature

# In[3]:


def maxTemperature(df,spark):
    return df.rdd.map(lambda d: d['temperature']).filter(lambda x: x != None).max()


# Please now do the same for the standard deviation of the temperature

# In[4]:


def sdTemperature(df,spark):
    return df.rdd.map(lambda d: d['temperature']).filter(lambda x: x != None).stdev()


# Please now do the same for the skew of the temperature. Since the SQL statement for this is a bit more complicated we've provided a skeleton for you. You have to insert custom code at four position in order to make the function work. Alternatively you can also remove everything and implement if on your own. Note that we are making use of two previously defined functions, so please make sure they are correct. Also note that we are making use of python's string formatting capabilitis where the results of the two function calls to "meanTemperature" and "sdTemperature" are inserted at the "%s" symbols in the SQL string.

# In[5]:


def skewTemperature(df,spark):    
#    return spark.sql("""
#SELECT 
#    (
#        1/##INSERT YOUR CODE HERE##
#    ) *
#    SUM (
#        POWER(##INSERT YOUR CODE HERE##-%s,3)/POWER(%s,3)
#    )
#
#as ##INSERT YOUR CODE HERE## from washing
#                    """ %(meanTemperature(df,spark),sdTemperature(df,spark)))##INSERT YOUR CODE HERE##
    temperatures = df.rdd.map(lambda d: d['temperature']).filter(lambda x: x != None)
    m = temperatures.mean()
    sd = temperatures.stdev()
    n = temperatures.count()
    return temperatures.map(lambda t: pow(t-m, 3)).sum()/n/pow(sd, 3)


# Kurtosis is the 4th statistical moment, so if you are smart you can make use of the code for skew which is the 3rd statistical moment. Actually only two things are different.

# In[6]:


def kurtosisTemperature(df,spark):    
    temperatures = df.rdd.map(lambda d: d['temperature']).filter(lambda x: x != None)
    print(temperatures.take(10))
    m = temperatures.mean()
    sd = temperatures.stdev()
    n = temperatures.count()
    return temperatures.map(lambda t: pow(t-m, 4)).sum()/n/pow(sd, 4)


# Just a hint. This can be solved easily using SQL as well, but as shown in the lecture also using RDDs.

# In[7]:


def correlation(rdd1, rdd2):
    mean1 = rdd1.mean()
    mean2 = rdd2.mean()
    sd1 = rdd1.stdev()
    sd2 = rdd2.stdev()
    rdd12 = rdd1.zip(rdd2)
    cov = rdd12.map(lambda r: (r[0]-mean1)*(r[1]-mean2)).sum()/float(rdd12.count())
    return cov/(sd1 * sd2)


# In[8]:


def correlationTemperatureHardness(df,spark):
    # data = df.rdd.map(lambda d: (d['temperature'], d['hardness'])).filter(lambda r: r[0] != None and r[1] == None)
    temperature = df.rdd.map(lambda d: d['temperature']).filter(lambda x: x != None)
    hardness = df.rdd.map(lambda d: d['hardness']).filter(lambda x: x != None)
    return correlation(temperature, hardness)


# In[9]:


def getTemperatureHardness(df,spark):
    return df.rdd.map(lambda d: [d['temperature'],d['hardness']]).filter(lambda r: r[0] != None)


# ### PLEASE DON'T REMOVE THIS BLOCK - THE FOLLOWING CODE IS NOT GRADED
# #axx
# ### PLEASE DON'T REMOVE THIS BLOCK - THE FOLLOWING CODE IS NOT GRADED

# Now it is time to connect to the cloudant database. Please have a look at the Video "Overview of end-to-end scenario" of Week 2 starting from 6:40 in order to learn how to obtain the credentials for the database. Please paste this credentials as strings into the below code
# 
# ### TODO Please provide your Cloudant credentials here

# In[10]:


### TODO Please provide your Cloudant credentials here by creating a connection to Cloudant and insert the code
### Please have a look at the latest video "Connect to Cloudant/CouchDB from ApacheSpark in Watson Studio" on https://www.youtube.com/c/RomeoKienzler
database = "washing" #as long as you didn't change this in the NodeRED flow the database name stays the same


# In[11]:


# @hidden_cell
credentials_1 = {

}


# In[12]:


#Please don't modify this function
def readDataFrameFromCloudant(database):
    cloudantdata=spark.read.load(database, "com.cloudant.spark")

    cloudantdata.createOrReplaceTempView("washing")
    spark.sql("SELECT * from washing").show()
    return cloudantdata


# In[13]:


spark = SparkSession    .builder    .appName("Cloudant Spark SQL Example in Python using temp tables")    .config("cloudant.host",credentials_1['custom_url'].split(':')[2].split('@')[1])    .config("cloudant.username", credentials_1['username'])    .config("cloudant.password",credentials_1['password'])    .config("jsonstore.rdd.partitions", 1)    .getOrCreate()


# In[14]:


df=readDataFrameFromCloudant(database)


# In[15]:


minTemperature(df,spark)


# In[16]:


meanTemperature(df,spark)


# In[17]:


maxTemperature(df,spark)


# In[18]:


sdTemperature(df,spark)


# In[19]:


skewTemperature(df,spark)


# In[20]:


kurtosisTemperature(df,spark)


# In[21]:


correlationTemperatureHardness(df,spark)


# In[22]:


getTemperatureHardness(df,spark)


# In[23]:


from pyspark.mllib.stat import Statistics

data = getTemperatureHardness(df,spark)
print(Statistics.corr(data))


# Congratulations, you are done, please download this notebook as python file using the export function and submit is to the gader using the filename "assignment3.1.py"
