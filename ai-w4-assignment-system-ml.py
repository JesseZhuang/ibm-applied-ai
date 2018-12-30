
# coding: utf-8

# # Assignment 4
# ## Understaning scaling of linear algebra operations on Apache Spark using Apache SystemML
# 
# In this assignment we want you to understand how to scale linear algebra operations from a single machine to multiple machines, memory and CPU cores using Apache SystemML. Therefore we want you to understand how to migrate from a numpy program to a SystemML DML program. Don't worry. We will give you a lot of hints. Finally, you won't need this knowledge anyways if you are sticking to Keras only, but once you go beyond that point you'll be happy to see what's going on behind the scenes. As usual, we run some import statements:

# In[1]:


get_ipython().system(u'pip install --upgrade systemml')


# In[2]:


from systemml import MLContext, dml
import numpy as np
import time


# Then we create an MLContext to interface with Apache SystemML. Note that we pass a SparkSession object as parameter so SystemML now knows how to talk to the Apache Spark cluster

# In[3]:


ml = MLContext(spark)


# Now we create some large random matrices to have numpy and SystemML crunch on it

# In[4]:


u = np.random.rand(1000,10000)
s = np.random.rand(10000,1000)
w = np.random.rand(1000,1000)


# Now we implement a short one-liner to define a very simple linear algebra operation
# 
# In case you are not familiar with matrix-matrix multiplication: https://en.wikipedia.org/wiki/Matrix_multiplication
# 
# sum(U' * (W . (U * S)))
# 
# 
# | Legend        |            |   
# | ------------- |-------------| 
# | '      | transpose of a matrix | 
# | * | matrix-matrix multiplication      |  
# | . | scalar multiplication      |   
# 
# 

# In[5]:


start = time.time()
res = np.sum(u.T.dot(w * u.dot(s)))
print (time.time()-start)


# As you can see this executes perfectly fine. Note that this is even a very efficient execution because numpy uses a C/C++ backend which is known for it's performance. But what happens if U, S or W get such big that the available main memory cannot cope with it? Let's give it a try:

# In[6]:


u = np.random.rand(10000,100000)
s = np.random.rand(100000,10000)
w = np.random.rand(10000,10000)


# After a short while you should see a memory error. This is because the operating system process was not able to allocate enough memory for storing the numpy array on the heap. Now it's time to re-implement the very same operations as DML in SystemML, and this is your task. Just replace all ###your_code_goes_here sections with proper code, please consider the following table which contains all DML syntax you need:
# 
# | Syntax        |            |   
# | ------------- |-------------| 
# | t(M)      | transpose of a matrix, where M is the matrix | 
# | %*% | matrix-matrix multiplication      |  
# | * | scalar multiplication      |   
# 
# ## Task

# In[7]:


script = """
res = sum(t(U) %*% (W * (U %*% S)))
"""


# To get consistent results we switch from a random matrix initialization to something deterministic

# In[8]:


u = np.arange(100000).reshape((100, 1000))
s = np.arange(100000).reshape((1000, 100))
w = np.arange(10000).reshape((100, 100))


# In[9]:


prog = dml(script).input('U', u).input('S', s).input('W', w).output('res')
res = ml.execute(prog).get('res')
print (res)


# If everything runs fine you should get *1.25260525922e+28* as result. Feel free to submit your DML script to the grader now!
# 
# ### Submission

# In[10]:


get_ipython().system(u'rm -f rklib.py')
get_ipython().system(u'wget https://raw.githubusercontent.com/romeokienzler/developerWorks/master/coursera/ai/rklib.py')


# In[11]:


from rklib import submit
key = "esRk7vn-Eeej-BLTuYzd0g"
part = "fUxc8"

email = ""
secret = ""
submit(email, secret, key, part, [part], script)

