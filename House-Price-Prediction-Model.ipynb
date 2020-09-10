## This notebook is implemented in kaggle kernel.

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

import seaborn as sns
import warnings 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import sklearn.metrics as metrics
import math

df_train = pd.read_csv(r'../input/train-house/Train.csv')
df_test = pd.read_csv(r'../input/test-house/Test.csv')


#Creating a copy of the train and test datasets
test_copy  = df_test.copy()
train_copy  = df_train.copy()


test_copy.head(3)

test_copy.head(3)

##Concat Train and Test datasets
train_copy['train']  = 1
test_copy['train']  = 0
df = pd.concat([train_copy, test_copy], axis=0,sort=False)

## Select object data type columns. 
object_columns_train = df.select_dtypes(include=['object'])

## Select numeric data type columns.
numerical_columns_train =df.select_dtypes(exclude=['object'])


## Fill missing values in society column with 'None'.
object_columns_train['society'] = object_columns_train['society'].fillna('None')


## Fill missing values in size & location column with most frequent value.
cols = ['size', 'location']
object_columns_train[cols] = object_columns_train[cols].fillna(object_columns_train.mode().iloc[0])


## FIll missing values in bath column with its median.
median = numerical_columns_train['bath'].median()
numerical_columns_train['bath'] = numerical_columns_train['bath'].fillna(median)


## Fill missing values in balcony column with 0
numerical_columns_train['balcony'] = numerical_columns_train['balcony'].fillna(0)


## Remove availability column from dataset due to low variance.
object_columns_train.drop(['availability'], axis=1, inplace=True)

#Using One hot encoder on categorical variables 
object_columns_train = pd.get_dummies(object_columns_train, columns= object_columns_train.columns)

object_columns_train.head(3)

## Concat Categorical (after encoding) and numerical features
df_final = pd.concat([object_columns_train, numerical_columns_train], axis=1,sort=False)
df_final.head()

df_final.shape

## Drop price column which has to be predicted.
df_train = df_final[df_final['train'] == 1]
df_train = df_train.drop(['train',],axis=1)

df_test = df_final[df_final['train'] == 0]
df_test = df_test.drop(['price'],axis=1)
df_test = df_test.drop(['train',],axis=1)

## XGB Regressor

##Separate Train and Targets and use logarithmetic value of price.
target= np.log(df_train['price'])
df_train.drop(['price'],axis=1, inplace=True)

df_final.shape

x_train,x_test,y_train,y_test = train_test_split(df_train, target, test_size=0.3,random_state=15)

xgb =XGBRegressor( booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.6, gamma=0,
             importance_type='gain', learning_rate=0.01, max_delta_step=0,
             max_depth=4, min_child_weight=1.5, n_estimators=2000,
             n_jobs=1, nthread=None, objective='reg:linear',
             reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1, 
             silent=None, subsample=0.8, verbosity=1, tree_method='gpu_hist')

#Fitting
xgb.fit(x_train, y_train)

## Prediction
predict1 = xgb.predict(x_test)

## Error check
print('Root Mean Square Error test (XGB)= ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict1))))

## Fit with all dataset in XGB Regressor
xgb.fit(df_train, target)

## Prediction on test dataset.
predict3 = xgb.predict(df_test)

# ## LGB Regressor

df_final_lgb = df_train.copy()
target_lgb = target.copy()

import re
df_final_lgb = df_final_lgb.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

x_train1,x_test1,y_train1,y_test1 = train_test_split(df_final_lgb, target_lgb, test_size=0.3,random_state=15)

lgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=12000,
                                       max_bin=200,
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.4, 
                                       device= 'gpu',
                                       gpu_platform_id= 0,
                                       gpu_device_id=0
                                       )

#Fitting
lgbm.fit(x_train1, y_train1, eval_metric='rmse')

## Prediction
predict = lgbm.predict(x_test)

print('Root Mean Square Error test (LGB) = ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict))))

lgbm.fit(df_final_lgb, target_lgb, eval_metric='rmse')

## Fitting With all the dataset
predict4 = lgbm.predict(df_test)

## Ensemble prediction
predict_y = ( predict3*0.45 + predict4 * 0.55)

## Make a submission file of predicted price.
submission = pd.DataFrame({
        "Price": predict_y
    })
submission.to_csv('submission.csv', index=False)
