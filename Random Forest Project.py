## RANDOM FOREST PROJECT ##

##For this project I will be exploring publicly available data from LendingClub.com. 
##Lending Club connects people who need money with people who have money.
##Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back.
## In this case I will try to create a model that will help predict this. ## 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

## Using Jupyter Notebook ## 

## Downlaoded a csv on loands ## 
loans = pd.read_csv('loan_data.csv')

loans.describe()
loans.head()

##EDA ## 
## I will get the credit score of the borrower for each policy credit outcome ## 
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha =0.5, color = 'blue', bins = 30, label = 'Credit Policy =1')
loans[loans['credit.policy']==0]['fico'].hist(alpha =0.5, color = 'red', bins = 30, label = 'Credit Policy =0')
plt.legend()
plt.xlabel('FICO')

##Creating a similar figure, except this time selecting by the not.fully.paid column.##
plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha =0.5, color = 'blue', bins = 30, label = 'not fully paid =1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha =0.5, color = 'red', bins = 30, label = 'not fully paid =0')
plt.legend()
plt.xlabel('FICO')

## Using Seaborn to produce a countplot and jointplot to help see the data in a better way. ## 
plt.figure(figsize=(11,7))
sns.countplot(loans['purpose'], hue = loans['not.fully.paid'])

sns.jointplot(x = loans['fico'], y = loans['int.rate'])


plt.figure(figsize=(11,7))
sns.lmplot(y = 'int.rate', x = 'fico', 
           data = loans, hue = 'credit.policy', col = 'not.fully.paid')

## I just remembered that the purpose column is categorical this means I need to transform it using dummy variables so sklearn will be able to understand them. ##

final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)

final_data.info()

## Now  I will split the data using Sklean.model_selection ## 
from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid', axis = 1)
y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) ## Everything but @Not Filly paid for X) ## 


## Now to train the decision tree ## 
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

##Fitting the data ## 
dtree.fit(X_train,y_train)

##Now I will Create predictions from the test set and create a classification report and a confusion matrix.##
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
##The resulst aren't too pretty, and so hopefully I can gain some more accuracy in te confusion matrix through a random forest##

## We can see the weighted average f1-score could possibly be better here too. ## 

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

##Predicting off the X_test Data 
rfc_predictions = rfc.predict(X_test)

## Results ##
print(confusion_matrix(y_test, rfc_predictions))
print(classification_report(y_test, rfc_predictions))

# In this case it really depends what metric I was trying to optimize for. 
# Notice the recall for each class for the models.
## My recall did better in the decision tree ##
# Neither did extremely well. ##
##Overall slighly better was Random Forest ##





















