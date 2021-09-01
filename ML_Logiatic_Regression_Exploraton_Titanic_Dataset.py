import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train = pd.read_csv('titanic_train.csv')

train.head()

##Find Null Values. Cabin looks to contain a few, (Noted)## 
train.isnull()


## Visualise exactly where, in this case (I knew Cabin contained null, but interestigly so does Age...) ##
##Yticks will make it clearer to give a real "Birds Eye View" of the data. ## 
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


sns.set_style('whitegrid')

## Classification Problem##
sns.countplot(x = 'Survived', data = train)
## Less People Survived ##

## Classification Problem##
sns.countplot(x = 'Survived', hue = 'Sex', data = train)
## Less Males Survived ##

## Classification Problem##
sns.countplot(x = 'Survived', hue = 'Pclass', data = train)
## Less 3rd class Survived ##

sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)

sns.countplot(x='SibSp',data=train)

train['Fare'].hist(color='green',bins=40,figsize=(8,4)) ##Most Passengers were in the cheaper Fare Class## #1912##

import cufflinks as cf 

cf.go_offline()

train['Fare'].iplot(kind='hist',bins=30)

sns.boxplot(x = 'Pclass', y = 'Age', data = train)
plt.figure(figsize =(10,7))

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
		
	## As I fund out through the heatmap age contains many nulll Values and so:
##I want to define the nulls along with theri respective Average at that Class. ##



   ## def impute_age(cols):
   ## Age = cols[0]
   ## Pclass = cols[1]
    
  ##  if pd.isnull(Age):

      ##  if Pclass == 1:
     ##       return 37
         ##This is Average for 1 whch can be seen in Boxplot ##

     ##   elif Pclass == 2:
      ##      return 29

     ##   else:
       ##     return 24
        ## No. 3 ## 
##else:
  ##      return Age
    ## Since they have an age ##

train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis = 1)

	sns.heatmap(train.isnull(),yticklabels=False, cbar= False) ##Missing Age Column fixed with Avg Age for Class##
	
	
	train.drop('Cabin', axis = 1, inplace = True) ##Get rid of Cabin##
	
train.head()


sns.heatmap(train.isnull(),yticklabels=False, cbar= False) ## No cabin ##

## want to drop all nulls however ##
train.dropna(inplace = True) 

##ML Model can't use Categorical Data so I will make a dummy variable to convert strings to a numerical variable. ##

pd.get_dummies(train['Sex'])
## Now boolean values for male and female##
##However, this may hinder the accuracy of my algorithim... This 
##can be known as In statistics, multicollinearity... a phenomenon in which one predictor variable in a multiple
##regression model can be linearly predicted from the others with a substantial degree of accuracy.
##So, I need to fix this by passing in a DF and drop column


pd.get_dummies(train['Sex'], drop_first = True)

sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True) -- Can see with Head() That neither columns are perfect predictors 
embark.head()


train = pd.concat([train, sex, embark], axis =1 )

##Need to drop columns not in use. ##


train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace = True)


train.head() ##Perfect for ML algorithim, all numerical. 

train.drop('PassengerId', axis = 1, inplace = 'True') ## The ID Was just an index after inspection ##

x= train.drop('Survived', axis =1 )
y= train['Survived']

from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions)) -  Here  I can check the precision










