#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv')


# In[3]:


df


# ### Question 1: Version of Numpy

# In[4]:


np.__version__


# ### Question 2: Number of records in the dataset

# In[5]:


len(df)


# ### Question 3: Most popular car manufacturers

# In[6]:


df['Make'].value_counts().nlargest(3)


# ### Question 4: Number of unique Audi car models

# In[8]:


df.loc[df['Make'] == 'Audi', 'Model'].nunique()


# ### Question 5: Number of columns with missing values

# In[9]:


sum(df.isna().sum(axis = 0) > 0)


# ### Question 6: Does the median value change after filling missing values

# In[11]:


previous_engine_cylinder_median = df['Engine Cylinders'].median()
engine_cylinder_mode = df['Engine Cylinders'].mode()
df['Engine Cylinders'].fillna(engine_cylinder_mode, inplace=True)
current_engine_cylinder_median = df['Engine Cylinders'].median()
previous_engine_cylinder_median != current_engine_cylinder_median


# ### Question 7: Value of the first element of w

# In[12]:


lotus_df = df.loc[df['Make'] == 'Lotus', ['Engine HP', 'Engine Cylinders']]
lotus_df_without_duplicates = lotus_df.drop_duplicates()
X = lotus_df_without_duplicates.values
XTX = X.T.dot(X)
XTX_inv = np.linalg.inv(XTX)
y = [1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800]
w = XTX_inv.dot(X.T).dot(y)
w[0]

