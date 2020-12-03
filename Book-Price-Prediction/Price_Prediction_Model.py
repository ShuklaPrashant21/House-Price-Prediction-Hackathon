import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings

# Load data 
df_train = pd.read_excel(r'C:\Users\lenovo\Downloads\DS_Practice\MACHINEHACK HACKATHONS\Book-Price-MachineHack\Data\Data_Train.xlsx')

df_test = pd.read_excel(r'C:\Users\lenovo\Downloads\DS_Practice\MACHINEHACK HACKATHONS\Book-Price-MachineHack\Data\Data_Test.xlsx')

df_train.head(3)

# Summary of data i.e datatype of columns
df_train.info()
df_test.info()

#Creating a copy of the train and test datasets
test_copy  = df_test.copy()
train_copy  = df_train.copy()

## Data Pre Processing

##Concat Train and Test datasets
train_copy['train']  = 1
test_copy['train']  = 0
df = pd.concat([train_copy, test_copy], axis=0,sort=False)

## Remove title column from data set because it is not necessary in model development.
df.drop(columns ='Title', axis=1, inplace =True)

df.drop(columns=['Synopsis', 'Author','Genre','Edition'], axis=1, inplace=True)

## Feature Engineering

## Extract Reviews and Ratings in numeric datatype.

df['Reviews'] = df.Reviews.apply(lambda r: float(r.split()[0]))
df['Ratings']= df.Ratings.str.extract('(\d+)')
df["Ratings"] = df.Ratings.astype(float)
df.head()

## Drop outliers in price column 
df.drop(df['Price'] > 9000, axis=0, inplace=True)

#Using One hot encoder on categorical variables 
dum_cat =pd.get_dummies(df['BookCategory'], drop_first = True)
dum_cat.shape

df_final = pd.concat([dum_cat, df], axis=1,sort=False)

## Import machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df_final.columns
## Drop BookCategory column
df_final.drop('BookCategory', axis=1, inplace=True)

## Drop price column which has to be predicted.
df_train2 = df_final[df_final['train'] == 1]
df_train2 = df_train2.drop(['train',],axis=1)

df_test2 = df_final[df_final['train'] == 0]
df_test2 = df_test2.drop(['Price'],axis=1)
df_test2 = df_test2.drop(['train',],axis=1)


##Separate Train and Targets and use logarithmetic value of price.
target =  np.log(df_train2['Price'])
df_train2.drop(['Price'],axis=1, inplace=True)

## Linear Regression Model
x_train,x_test,y_train,y_test = train_test_split(df_train2, target, test_size=0.3,random_state=5)

## Setting intercept as true
lgr = LinearRegression(fit_intercept =True)

## MODEL
model_lin1 = lgr.fit(x_train, y_train)

## Predicting model on test set
price_predictions_lin1 = lgr.predict(x_test)

## Computing MSE and RMSE
lin_mse1 = mean_squared_error(y_test, price_predictions_lin1)
lin_rmse1 = np.sqrt(lin_mse1)
print(lin_rmse1)

## R squared value
r2_lin_test1 = model_lin1.score(x_test, y_test)
r2_lin_train1 = model_lin1.score(x_train, y_train)
print(r2_lin_test1, r2_lin_train1)

## Regression diagnostics :- Resident plot analysis
## It is differnce test data and your prediction. It is just difference between actual & predicted value.
residuals1 = y_test - price_predictions_lin1
sns.regplot(x = price_predictions_lin1, y=residuals1, scatter=True, fit_reg=False, data=df_final)
residuals1.describe()

df1 = pd.DataFrame({'Actual': y_test, 'Predicted':price_predictions_lin1})
df1.head(10)

## Graphical Representation of predicted price and actual price
df1.plot(kind='line',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.show()

## Predict on test dataset
lgr_test = lgr.fit(df_train2, target)
predict_test1 = lgr_test.predict(df_test2)

## Random Forest Regression 

## MODEL PARAMETERS
rf = RandomForestRegressor(n_estimators = 100, max_features='auto', max_depth=100, min_samples_split=10, min_samples_leaf=4, random_state=3)

## MODEL
model_rf1 =rf.fit(x_train, y_train)
## Predicting model on test set
salary_predictions_rf1 = rf.predict(x_test)

## Computing MSE and RSME
rf_mse1 = mean_squared_error(y_test, salary_predictions_rf1)
rf_rmse1 = np.sqrt(rf_mse1)
print(rf_rmse1)

## R Squared value
r2_rf_test1 = model_rf1.score(x_test, y_test)
r2_rf_train1 = model_rf1.score(x_train, y_train)
print(r2_rf_test1, r2_rf_train1)

df2 = pd.DataFrame({'Actual': y_test, 'Predicted':salary_predictions_rf1})
df2.head(10)

## Graphical Representation of predicted price and actual price
df2.plot(kind='line',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.show()

## Predict on whole test dataset
rf_test = rf.fit(df_train2, target)
predict_test2 = rf_test.predict(df_test2)

## Ensemble Prediction Technique

predict_price = (predict_test1 *0.45 + predict_test2 *0.55)

## Make a submission file of predicted price.
submission = pd.DataFrame({
        "Price": predict_test2
    })
submission.to_csv('submission.csv')
