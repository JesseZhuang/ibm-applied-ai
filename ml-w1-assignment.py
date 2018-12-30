
# coding: utf-8

# This is the first assgiment for the Coursera course "Advanced Machine Learning and Signal Processing"
# 
# The first step is to insert the credentials to the Apache CouchDB / Cloudant database where your sensor data ist stored to. 
# 
# 1. In the project's overview tab of this project just select "Add to project"->Connection
# 2. From the section "Your service instances in IBM Cloud" select your cloudant database and click on "create"
# 3. Now click in the empty cell below labeled with "your cloudant credentials go here"
# 4. Click on the "10-01" symbol top right and selecrt the "Connections" tab
# 5. Find your data base connection and click on "Insert to code"
# 
# The following video illustrates this process: https://www.youtube.com/watch?v=dCawUGv7qgs
# 
# Done, just execute all cells one after the other and you are done - just note that in the last one you have to update your email address (the one you've used for coursera) and obtain a submittion token, you get this from the programming assingment directly on coursera.

# In[7]:


#your cloudant credentials go here
# @hidden_cell
credentials_1 = {

}


# In[8]:


spark = SparkSession    .builder    .appName("Cloudant Spark SQL Example in Python using temp tables")    .config("cloudant.host",credentials_1['custom_url'].split('@')[1])    .config("cloudant.username", credentials_1['username'])    .config("cloudant.password",credentials_1['password'])    .config("jsonstore.rdd.partitions", 1)    .getOrCreate()


# In[9]:


df=spark.read.load('shake', "com.cloudant.spark")

df.createOrReplaceTempView("df")
spark.sql("SELECT * from df").show()


# In[11]:


get_ipython().system(u'rm -Rf a2_m1.parquet')


# In[12]:


df = df.repartition(1)
df.write.json('a2_m1.json')


# In[13]:


get_ipython().system(u'rm -f rklib.py')
get_ipython().system(u'wget https://raw.githubusercontent.com/romeokienzler/developerWorks/master/coursera/ai/rklib.py')


# In[14]:


get_ipython().system(u'zip -r a2_m1.json.zip a2_m1.json')


# In[16]:


get_ipython().system(u'base64 a2_m1.json.zip > a2_m1.json.zip.base64')


# In[17]:


from rklib import submit
key = "1injH2F0EeiLlRJ3eJKoXA"
part = "wNLDt"
email = ""
secret = ""

with open('a2_m1.json.zip.base64', 'r') as myfile:
    data=myfile.read()
submit(email, secret, key, part, [part], data)

