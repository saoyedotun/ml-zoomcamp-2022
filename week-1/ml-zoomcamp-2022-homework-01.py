#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv')


# ### Question 1
# What's the version of NumPy that you installed?
# 
# You can get the version information using the version field:

# In[3]:


np.__version__


# ### Question 2
# How many records are in the dataset?
# 
# Here you need to specify the number of rows.
# 
# - 16
# - 6572
# - 11914
# - 18990

# In[4]:


df.shape[0]


# ### Question 3
# Who are the most popular car manufacturers (top-3) according to the dataset?
# 
# - Chevrolet, Volkswagen, Toyota
# - Chevrolet, Ford, Toyota
# - Ford, Volkswagen, Toyota
# - Chevrolet, Ford, Volkswagen

# In[5]:


df['Make'].value_counts()[:3]


# ### Question 4
# What's the number of unique Audi car models in the dataset?
# 
# - 3
# - 16
# - 26
# - 34

# In[6]:


df[df['Make'] =='Audi']['Model'].nunique()


# ### Question 5
# How many columns in the dataset have missing values?
# 
# - 5
# - 6
# - 7
# - 8

# In[9]:


nan_columns = df.columns[df.isna().any()].to_list()
print(f"Columns with missing values: {nan_columns}")
print(f"Number of columns with missing values = {len(nan_columns)}")


# ### Question 6
# Find the median value of "Engine Cylinders" column in the dataset.
# 
# Next, calculate the most frequent value of the same "Engine Cylinders".
# 
# Use the fillna method to fill the missing values in "Engine Cylinders" with the most frequent value from the previous step.
# 
# Now, calculate the median value of "Engine Cylinders" once again.
# 
# Has it changed?

# In[10]:


df['Engine Cylinders'].median()


# In[11]:


df['Engine Cylinders'].mode()


# In[12]:


df['Engine Cylinders'].fillna(float(df['Engine Cylinders'].mode()), inplace=True)


# In[13]:


df['Engine Cylinders'].isna().value_counts()


# In[14]:


df['Engine Cylinders'].median()


# ### Question 7
# Select all the "Lotus" cars from the dataset.
# 
# Select only columns "Engine HP", "Engine Cylinders".
# 
# Now drop all duplicated rows using drop_duplicates method (you should get a dataframe with 9 rows).
# 
# Get the underlying NumPy array. Let's call it X.
# 
# Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
# 
# Invert XTX.
# 
# Create an array y with values [1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800].
# 
# Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
# 
# What's the value of the first element of w?
# 
# - -0.0723
# - 4.5949
# - 31.6537
# - 63.5643

# In[15]:


lotus_cars = df[df['Make'] =='Lotus']


# In[16]:


selected_cols = lotus_cars[['Engine HP', 'Engine Cylinders']]


# In[19]:


selected_cols = selected_cols.drop_duplicates()
selected_cols.shape[0]


# In[20]:


X = selected_cols.to_numpy()
X


# In[21]:


XT = X.T
XT


# In[22]:


XTX = np.dot(XT, X)
XTX


# In[23]:


XTXI = np.linalg.inv(XTX)
XTXI


# In[24]:


y = np.array([1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800])


# In[25]:


w = np.dot(np.dot(XTXI, XT), y)


# In[26]:


w[0]

