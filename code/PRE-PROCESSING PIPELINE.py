#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


# In[11]:


input_data = pd.read_csv('house_data.csv')

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def preprocess_data(input_data):
    # Separate numerical and categorical columns
    numerical_cols = input_data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = input_data.select_dtypes(include=['object']).columns

    # Pipeline for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
        ('scaler', StandardScaler())  # Standardize numerical features
    ])

    # Pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent value
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical variables
    ])

    # Bundle transformers for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Apply the preprocessing pipeline to the input data
    processed_data = pd.DataFrame(preprocessor.fit_transform(input_data))

    return processed_data

# Example usage:
# Assuming 'input_data' is your original DataFrame
processed_data = preprocess_data(input_data)


# In[6]:


processed_data


# In[12]:


print("pandas version:", pd.__version__)


# In[15]:


get_ipython().system('pip show pandas')


# In[16]:


get_ipython().system('pip show scikit-learn')


# In[ ]:




