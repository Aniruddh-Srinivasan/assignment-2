import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import os
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn import metrics

print(os.getcwd())

## before working please ensure this file and data set are in same directory
## reading the selected features columns from the data set
tab1=pd.read_csv('train logis.csv',usecols=['Pclass','Age','Sex','SibSp','Parch','Embarked','Survived'])


## Encoding the categr
label_X = LabelEncoder()
df2=tab1['Sex']
tab1['Sex']=label_X.fit_transform(df2)

label_X = LabelEncoder()
df2=tab1['Embarked']
tab1['Embarked']=label_X.fit_transform(df2)

mean1=tab1['Age'].mean()
tab1.loc[tab1['Age'].isna(),['Age']]=mean1

mean1=tab1['Embarked'].mode()
tab1.loc[tab1['Embarked'].isna(),['Embarked']]=mean1
 
##Pre Processing done

## Model building and confusion matrix for accuracy.
X=tab1.iloc[:,[1,2,3,4,5,6]]
Y=tab1.iloc[:,[0]]
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
logis=LogisticRegression()
model=logis.fit(x_train,y_train)
y_predict=model.predict(x_test)
cm = metrics.confusion_matrix(y_test,y_predict)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


##predict the test data
tab2=pd.read_csv('test logis.csv',usecols=['Pclass','Age','Sex','SibSp','Parch','Embarked'])

print(tab2)

label_X = LabelEncoder()
df2=tab2['Sex']
tab2['Sex']=label_X.fit_transform(df2)

label_X = LabelEncoder()
df2=tab2['Embarked']
tab2['Embarked']=label_X.fit_transform(df2)

mean1=tab2['Age'].mean()
tab2.loc[tab2['Age'].isna(),['Age']]=mean1

mean1=tab2['Embarked'].mode()
tab2.loc[tab2['Embarked'].isna(),['Embarked']]=mean1
x_test=tab2.iloc[:,[0,1,2,3,4,5]]
print(x_test)
y_predict=model.predict(x_test)


print(y_predict)
tab2.insert(1,column="Survived",value=y_predict)
print(tab2['Survived'].value_counts())

#tab2.to_csv('Test_Prediction.csv')




