#!/usr/bin/env python
# coding: utf-8

# #### Course ML Zoomcamp Homework 1

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv')


# In[3]:


data.head()


# <b>1. What's the version of NumPy that you installed?</b>

# In[4]:


np.__version__


# <b>2. How many records are in the dataset?</b>

# In[5]:


data.shape


# In[8]:


data.shape[0]    # Returns number of rows


# In[11]:


len(data)    # Returns number of rows


# In[9]:


data.shape[1]    # Returns number of columns


# <b>3. Who are the most frequent car manufacturers (top-3) according to the dataset?</b>

# In[12]:


data['Make'].value_counts().nlargest(3)


# In[13]:


data['Make'].value_counts()


# <b>4. What's the number of unique Audi car models in the dataset?</b>

# In[32]:


unique_audi_car_models_number = data[(data['Make']=='Audi')]['Model'].nunique()
print(unique_audi_car_models_number)


# <b>5. How many columns in the dataset have missing values?</b>

# In[15]:


data.info()


#     5 columns

# <b>6.</b>
# - Find the median value of "Engine Cylinders" column in the dataset.
# - Next, calculate the most frequent value of the same "Engine Cylinders".
# - Use the fillna method to fill the missing values in "Engine Cylinders" with the most frequent value from the previous step.
# - Now, calculate the median value of "Engine Cylinders" once again.
# - Has it changed?

# In[24]:


engine_cylinder_median = data['Engine Cylinders'].median()
print(engine_cylinder_median)


# In[21]:


engine_cylinder_mode = data['Engine Cylinders'].mode()
print(engine_cylinder_mode)


# In[25]:


engine_cylinder_missing_value_mode = data['Engine Cylinders'].fillna(value=engine_cylinder_mode)
print(engine_cylinder_missing_value_mode)


# In[28]:


new_median_engine_cylinder = engine_cylinder_missing_value_mode.median()
print(new_median_engine_cylinder)


# In[30]:


engine_cylinder_missing_value_mode.isnull().sum()


# <b>7.</b>
# - Select all the "Lotus" cars from the dataset.
# - Select only columns "Engine HP", "Engine Cylinders".
# - Now drop all duplicated rows using drop_duplicates method (you should get a dataframe with 9 rows).
# - Get the underlying NumPy array. Let's call it X.
# - Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
# - Invert XTX.
# - Create an array y with values [1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800].
# - Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
# - What's the value of the first element of w?

# In[34]:


lotus_cars = data_new=data[data["Make"]=="Lotus"]
# print(lotus_cars)


# In[36]:


lotus_cars_selected_cols = lotus_cars[["Engine HP", "Engine Cylinders"]]
lotus_cars_selected_cols


# In[37]:


lotus_cars_selected_cols.duplicated().any()


# In[38]:


len(lotus_cars_selected_cols.duplicated())


# In[39]:


data_drop_duplicates = lotus_cars_selected_cols.drop_duplicates()
data_drop_duplicates


# In[41]:


X = data_drop_duplicates.iloc[:,:].values


# In[42]:


X[:,:]


# In[46]:


XTX = np.dot(X.T,X)
XTX


# In[47]:


XTX.shape


# In[48]:


y = np.array([1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800])


# In[49]:


X_inv = np.linalg.inv(XTX)


# In[51]:


Y = np.dot(X_inv, X.T)


# In[54]:


w = np.dot(Y, y)
w

