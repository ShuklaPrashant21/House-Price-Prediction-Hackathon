#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[3]:


df_train = pd.read_excel(r'C:\Users\lenovo\Downloads\DS_Practice\MACHINEHACK HACKATHONS\Book-Price-MachineHack\Data\Data_Train.xlsx')


# In[4]:


df_test = pd.read_excel(r'C:\Users\lenovo\Downloads\DS_Practice\MACHINEHACK HACKATHONS\Book-Price-MachineHack\Data\Data_Test.xlsx')


# In[5]:


df_train.head(3)


# In[6]:


df_train.info()


# In[7]:


df_test.info()


# ## Data Exploration

# ### Analyse Price of book column.

# Let's proceed and check the distribution of the target variable.

# In[8]:


#SalePrice
sns.distplot(df_train['Price'])


# In[9]:


#skewness
print("The skewness of SalePrice is {}".format(df_train['Price'].skew()))


# Let's log transform this variable and see if this variable distribution can get any closer to normal.

# In[10]:


#now transforming the target variable
target = np.log(df_train['Price'])
print('Skewness is', target.skew())
sns.distplot(target)


# ### Analyse Genre column in train data set.

# In[11]:


df_train.Genre.value_counts()


# In[12]:


sns.countplot(x='Genre', data=df_train)
plt.xticks(rotation='vertical')
plt.show()


# ### Analyse BookCategory column in train data set.

# In[13]:


df_train.BookCategory.value_counts()


# In[14]:


sns.countplot(x='BookCategory', data=df_train)
plt.xticks(rotation='vertical')
plt.show()


# In[15]:


## Box and whiskers plot is very useful to find relationship between numerical & categorical variables

sns.boxplot(x='BookCategory', y = 'Price', data=df_train)
plt.xticks(rotation = 'vertical')
plt.show()


# In[16]:


## Cricket movie :- Fire In Babylon


# In[17]:


df_train.Title.value_counts()


# ## Data Pre Processing 

# In[18]:


#Creating a copy of the train and test datasets
test_copy  = df_test.copy()
train_copy  = df_train.copy()


# In[19]:


##Concat Train and Test datasets
train_copy['train']  = 1
test_copy['train']  = 0
df = pd.concat([train_copy, test_copy], axis=0,sort=False)


# In[20]:


df.info()


# In[21]:


## Remove title column from data set because it is not necessary in model development.
df.drop(columns ='Title', axis=1, inplace =True)


# In[22]:


## Extract reviews in numeric form 
df['Reviews(out_of_5)'] = df['Reviews'].str[:3]
## Make column numeric
df['Reviews(out_of_5)'] = pd.to_numeric(df["Reviews(out_of_5)"]) 
## drop reviews column
df.drop(['Reviews'],axis=1, inplace=True)


# In[23]:


## Extract ratings in numeric form 
df['Ratings_count'] = df['Ratings'].str[:2]


# In[24]:


import re

df['Ratings_count'] = df['Ratings'].str.replace(r'[^\d.]+', '')


# In[25]:


## Make column data type numeric
df['Ratings_count'] = pd.to_numeric(df['Ratings_count'])

## Drop Ratings column
df.drop(['Ratings'], axis=1, inplace=True)


# In[26]:


df.info()


# In[45]:


df_author = df['Author'].value_counts()>50
df_author


# In[39]:


## Drop price column which has to be predicted.
df_train = df[df['train'] == 1]
df_train = df.drop(['train',],axis=1)

df_test = df[df['train'] == 0]
df_test = df_test.drop(['Price'],axis=1)
df_test = df_test.drop(['train',],axis=1)


# In[46]:


##Separate Train and Targets and use logarithmetic value of price.
target= np.log(df_train['Price'])
df_train.drop(['Price'],axis=1, inplace=True)


# In[47]:


df.shape


# In[ ]:





# In[ ]:





# #### Baseline Model

# In[27]:


## Use this after train and test split on y_test model.
from sklearn.metrics import mean_squared_error


# In[28]:


## Baseline Model
base_mean = np.mean(df_train['Price'])
print(base_mean)


# In[29]:


## Representing some value till length of test data
base_mean = np.repeat(base_mean, len(df_train['Price']))


# In[30]:


## Representing some value till length of test data
base_mean


# In[31]:


## Finding the RMSE(Root Mean Squared Error)
## RMSE computes the difference between the test value and the predicted value and squared them and divides them by number of samples.

base_root_mean_square_error = np.sqrt(mean_squared_error(df_train['Price'], base_mean))
print(base_root_mean_square_error)


# In[ ]:




