
# coding: utf-8

# ### standard deviation, 2nd statistical moment
# 
# $S_N = {\sqrt {\frac 1N\sum_{i=1}^N(x_i-\bar x)^2}}$
# 
# ### skewness, 3rd statistical moment
# How assymetrica data is spread around the mean.
# 
# $ skewness: \gamma_1 = \frac 1N\frac{\sum_{i=1}^N(x_i-\bar x)^3}{s^3}$ where $s$ is the standard deviation
# 
# ### kurtosis, 4th statistical moment
# Shape of skew.
# 
# $kurtosis = \frac 1N\frac{\sum_{i=1}^N(x_i-\bar x)^4}{s^4}$

# In[1]:


import statistics

list1 = [1,2,4,5,34,1,32,4,34,2,1,3]
print(list1)


# In[2]:


statistics.mean(list1)


# In[3]:


statistics.median(list1)


# In[4]:


list2 = [1,2,4,5,34,1,32,4,34,2,1,3]
statistics.median(list2)


# In[5]:


list3 = [34,1,23,4,3,3,12,4,3,1]
statistics.stdev(list3)


# In[6]:


lists = [range(100), [49]*100, [49]*100 + [100]]

def my_mean(l):
    return sum(l)/len(l)
    
mean = list(map(my_mean, lists))
print(mean)


# In[7]:


from math import sqrt

def my_stdev(l):
    m = my_mean(l)
    return sqrt(sum(list(map(lambda x: pow(x-m, 2), l)))/len(l))

stds = list(map(my_stdev, lists))
print(stds)


# In[8]:


def my_kurtosis(l):
    sigma = statistics.stdev(l)
    return 0 if sigma == 0 else sum(list(map(lambda x: pow(x-statistics.mean(l), 4), l)))/pow(statistics.stdev(l), 4)/len(l)


# In[9]:


print(list(map(my_kurtosis, lists)))


# In[10]:


print(my_kurtosis([34,1,23,4,3,3,12,4,3,1]))

