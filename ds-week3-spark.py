
# coding: utf-8

# # Exercise 2
# ## Part 1
# Now let's calculate covariance and correlation by ourselves using ApacheSpark
# 
# 1st we crate two random RDD’s, which shouldn't correlate at all.
# 

# In[1]:


import random
rddX = sc.parallelize(random.sample(range(100),100))
print rddX.take(10)
rddY = sc.parallelize(random.sample(range(100),100))


# Now we calculate the mean, note that we explicitly cast the denominator to float in order to obtain a float instead of int

# In[2]:


meanX = rddX.sum()/float(rddX.count())
meanY = rddY.sum()/float(rddY.count())
print meanX
print meanY


# Now we calculate the covariance

# In[3]:


rddXY = rddX.zip(rddY)
covXY = rddXY.map(lambda (x,y): (x-meanX)*(y-meanY)).sum()/float(rddXY.count())
covXY


# Covariance is not a normalized measure. Therefore we use it to calculate correlation. But before that we need to calculate the indivicual standard deviations first

# In[4]:


from math import sqrt
n = rddXY.count()
sdX = sqrt(rddX.map(lambda x : pow(x-meanX,2)).sum()/n)
sdY = sqrt(rddY.map(lambda x : pow(x-meanY,2)).sum()/n)
print(sdX)
print(sdY)


# Now we calculate the correlation

# In[5]:


corrXY = covXY / (sdX * sdY)
corrXY


# In[6]:


def covariance(rdd1, rdd2):
    mean1 = rdd1.sum()/float(rdd1.count())
    mean2 = rdd2.sum()/float(rdd2.count())
    rdd12 = rdd1.zip(rdd2)
    return rdd12.map(lambda (x,y): (x-mean1)*(y-mean2)).sum()/float(rdd12.count())

print covariance(rddX, rddY)


# In[9]:


def correlation(rdd1, rdd2):
    mean1 = rdd1.sum()/float(rdd1.count())
    mean2 = rdd2.sum()/float(rdd2.count())
    sd1 = sqrt(rdd1.map(lambda x : pow(x-mean1,2)).sum()/rdd1.count())
    sd2 = sqrt(rdd2.map(lambda x : pow(x-mean2,2)).sum()/rdd2.count())
    return covariance(rdd1, rdd2)/(sd1 * sd2)

print correlation(rddX, rddY)

def covariance_correlation(rdd1, rdd2):
    mean1 = rdd1.sum()/float(rdd1.count())
    mean2 = rdd2.sum()/float(rdd2.count())
    sd1 = sqrt(rdd1.map(lambda x : pow(x-mean1,2)).sum()/rdd1.count())
    sd2 = sqrt(rdd2.map(lambda x : pow(x-mean2,2)).sum()/rdd2.count())
    rdd12 = rdd1.zip(rdd2)
    cov = rdd12.map(lambda (x,y): (x-mean1)*(y-mean2)).sum()/float(rdd12.count())
    return cov, cov/(sd1 * sd2)

print covariance_correlation(rddX, rddY)


# ## Part 2
# No we want to create a correlation matrix out of the four RDDs used in the lecture

# In[10]:


from pyspark.mllib.stat import Statistics
import random
column1 = sc.parallelize(range(100))
column2 = sc.parallelize(range(100,200))
column3 = sc.parallelize(list(reversed(range(100))))
column4 = sc.parallelize(random.sample(range(100),100))
data = column1.zip(column2).zip(column3).zip(column4).map(lambda (((a,b),c),d) : (a,b,c,d) ).map(lambda (a,b,c,d) : [a,b,c,d])
print Statistics.corr(data)


# In[11]:


column5 = sc.parallelize([1,2,3,4,5,6,7,8,9,10])
print column5.take(10)
column6 = sc.parallelize([7,6,5,4,5,6,7,8,9,10])

data = column5.zip(column6).map(lambda (x,y):[x,y])
print Statistics.corr(data)


# In[12]:


print covariance(column5, column6)
print correlation(column5, column6)


# In[14]:


print covariance_correlation(sc.parallelize([1,2,3,4,5,6,7]), sc.parallelize([7,6,5,4,5,6,7]))

