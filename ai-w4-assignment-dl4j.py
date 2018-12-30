
# coding: utf-8

# # Sonar Example
# 
# In this Exercise you will, build a Neural Netowrk to Classify Sonar Readings as either a "Mine" or a "Rock"
# 
# ## Data Set Information:
# 
# The file "sonar.mines" contains 111 patterns obtained by bouncing sonar signals off a metal cylinder at various angles and under various conditions. The file "sonar.rocks" contains 97 patterns obtained from rocks under similar conditions. The transmitted sonar signal is a frequency-modulated chirp, rising in frequency. The data set contains signals obtained from a variety of different aspect angles, spanning 90 degrees for the cylinder and 180 degrees for the rock. 
# 
# Each pattern is a set of 60 numbers in the range 0.0 to 1.0. Each number represents the energy within a particular frequency band, integrated over a certain period of time. The integration aperture for higher frequencies occur later in time, since these frequencies are transmitted later during the chirp. 
# 
# The label associated with each record contains the letter "R" if the object is a rock and "M" if it is a mine (metal cylinder). The numbers in the labels are in increasing order of aspect angle, but they do not encode the angle directly.
# 
# 

# # Download the data
# 
# When this command completes you will have a file "sonar.all-data"
# 

# In[1]:


get_ipython().system(u'wget https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data')


# # Rename the DataFile
# 
# Rename the datafile to sonar.csv

# In[3]:


get_ipython().system(u'mv sonar.all-data sonar.csv')



# # Convert Text Labels to Integers
# 
# You will create a Keras Neural Network to classify each record as a Mine or a Rock. 
# 
# Although, It is straightforward to keep the labels "M" or "R" in Keras and have working code, the goal of this exercise is to save the model and then load the model into DeepLearning4J a java framework. The Java Code to import has been prebuilt and precompiled and expects numeric labels. With that restriction in mind, convert the "M's" to "0" and the "R's" to "1" with the following commands. 

# In[4]:


get_ipython().system(u"sed -i -e 's/M/0/g' sonar.csv")


# In[5]:


get_ipython().system(u"sed -i -e 's/R/1/g' sonar.csv")


# # Verify the contents of the file.
# 
# The file has 60 features per line, followed by a label of 0 or 1. 
# 
# The data is not shuffled, although for best neural network training performance shuffling would be advised. 
# 
# To verify that the above conversion succeeded view the head and the tail of the file. 

# In[6]:


get_ipython().system(u'head sonar.csv')


# In[7]:


get_ipython().system(u'tail sonar.csv')


# # Build a Neural Network
# 
# Build a Keras Neural Network to Process the data file. By training a Neural Network we are feeding the network the features and asking it to make a prediction of which class of object those readings are from. 
# 
# We will build a Feed Forward Neural Network using Keras Sequential Model. 
# 
# First some imports
# 

# In[1]:


import keras

keras.__version__
keras.__path__


# In[ ]:


#!pip --version
#!pip uninstall -y keras
# !rm -rf /gpfs/fs01/user/sabc-b3ac605152f5ac-c574be55bfaa/.local/lib/python2.7/site-packages/keras
#!pip install keras==2.2.0
# !ls /gpfs/fs01/user/sabc-b3ac605152f5ac-c574be55bfaa/.local/lib/python2.7/site-packages/keras


# In[3]:


import keras

keras.__version__


# In[4]:


import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils


# # Set Random Seed
# 
# 
# Neural Networks begin by defining a computation grid with random weights applied to each initial calculation. 
# 
# For repeatable results setting a random seed guarantees that the second run will be the same as the first.
# 
# 

# In[7]:


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# # Load the data into a pandas dataframe and Split into Features and Labels
# 
# The first 60 fields are measurements from the sonar, the last field is the Label
# 

# In[8]:


# load dataset
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
print(X.shape)


# # Encode Labels
# 
# The following code converts the Labels to integers, this section would actually work on the unmodified dataset containing "M" or "R" for labels, so in this case the step is redundant. 
# 
# The Code also takes the integers and converts to one-hot, or dummy encoding. 
# 
# Given n labels dummy encoding creates an array of length n.
# The array will have a "1" value corresponding to the label and all ther values will be "0"
# 
# For this example with 2 labels, dummy encoding will make the following conversion. 
# 
# Original Data
# 
# ```
# 0
# 1
# 0
# ```
# 
# Dummy Encoded
# 
# ```
# 1,0
# 0,1
# 1,0
# ```
# 
# To verify you can uncomment the line. 
# 
# ```
# print(dummy_y)
# ```
# 
#  
# 
# 

# In[9]:


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y)



# # Build a model
# 
# Your code here, in this case you are on your own to build a working Neural Network. 
# 
# You can review the Keras section for examples. 
# 
# You are free to decide the depth and features of the Neural Network. 
# 
# Note however, the first Layer has to have input_dim = 60 to correspond to the number of features and 
# the last layer has to have 2 nodes to correspond to the number of labels.
# 
# How will you know you have a good model? 
# 
# Accuracy levels of about .80 can be expected with this dataset.
# 
# 

# In[10]:


# create model
model = Sequential()
model.add(Dense(60,input_shape=(60,)))
model.add(Dense(64,activation='relu'))
model.add(Dense(2,activation='sigmoid'))


# # Compile the Model and Train
# 
# Modify the following cell and set your number of epochs and your batch size. 
# 
# Depending on your model it may train in 20 epochs or it may take 40, or it may not train at all. 
# 
# Replace the "***Your VALUE HERE**" with a numeric value. 
# 
# If your loss function is not decreasing then your model is not training, modify your model and try again. 
# 
# 
# 

# In[11]:



# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, dummy_y, epochs=50, batch_size=64)


# In[13]:


model.predict(X)


# # Save your Model
# 
# Your Model will be loaded into dl4j and run in a Spark context. A saved model includes the weights and the computation graph needed for either further training or inference. In this example we will load the model into dl4j and pass it our datafile and evaluate the accuracy of the model in dl4j running in spark. 
# 

# In[12]:


model.save('my_modelx.h5')


# # Verify your model has saved
# 
# The ls command should show your model in the local directory of this notebook. 

# In[13]:


get_ipython().system(u'ls *.h5')



# # Run your code in DL4J on Spark
# 
# 
# DL4J has a KerasModelImport feature. Java code has been written and compiled that will import a keras model, run the model on a spark cluster. 
# 
# You can view the code here.
# 
# https://github.com/maxpumperla/dl4j_coursera/blob/master/src/main/java/skymind/dsx/KerasImportCSVSparkRunner.java
# 
# This Jar has the compiled class. 
# 
# https://github.com/maxpumperla/dl4j_coursera/releases/download/v0.4/dl4j-snapshot.jar
# 
# 
# ###  
# 
# The class KerasImportCSVSparkRunner takes the following command line options, 
# 
# * indexLabel
#     * Column in the data file containing Labels
#     * Labels must be numeric
# * train
#     * Set to true or false
#     * true: perform training using provided data file
#     * false: perform evaluation using provided data file
# * numClasses 
#     * number of classes
# * modelFileName
#     * Saved h5 archive of your Keras Model
# * dataFileName 
#     * DataFile to run training/evaluation with
# 
# 
# 
# 
# 

# In[23]:


get_ipython().system(u'wget https://github.com/maxpumperla/dl4j_coursera/releases/download/v0.4/dl4j-snapshot.jar')


# # Run your code in Spark
# 
# The output from Spark is rather verbose, lots of notices and warnings. 
# 
# This code will take time. 
# 
# To verify success look for output similar to this at the end. 
# 
# ```
# 
# ==========================Scores========================================
#  # of classes:    2
#  Accuracy:        0.7933
#  Precision:       0.8064
#  Recall:          0.7855
#  F1 Score:        0.7514
# ========================================================================
# 
# ```

# In[14]:


get_ipython().system(u'echo $MASTER')
get_ipython().system(u'ls *.jar')


# In[15]:


import keras

keras.__version__


# In[16]:


get_ipython().system(u' $SPARK_HOME/bin/spark-submit  --class skymind.dsx.KerasImportCSVSparkRunner  --files sonar.csv,my_modelx.h5  --master $MASTER  dl4j-snapshot.jar    -batchSizePerWorker 15    -indexLabel 60    -train false    -numClasses 2    -modelFileName  my_modelx.h5 -dataFileName sonar.csv')


# # DONE 
# 
# Great Job !!!

# # Grading this exercise
# 
# In order to get a grade for this exercise please copy the value for "Accuracy" into the Grader. 
# 
# How to find the Accuracy of your model. 
# 
# When the model completes the Evaluation will be logged to the console. 
# 
# The lines will look like this. Note that values have been removed, in your output you will see numeric values in place of the "xxx"
# 
# ```
# ==========================Scores========================================
#  # of classes:    2
#  Accuracy:        0.xxxx
#  Precision:       0.xxxx
#  Recall:          0.xxxx
#  F1 Score:        0.xxxx
# ========================================================================
# ```
# 
# Copy the value of "Accuracy" into the grader to pass this programing assignment. 
# 
