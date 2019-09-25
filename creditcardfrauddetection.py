#!/usr/bin/env python
# coding: utf-8

# In[22]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn

print('Python: {}'.format(sys.version))


# In[23]:


#import pakages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[24]:


#load the dataset
data=pd.read_csv('F:\old\my videos\MACHINE LEARNING\Ml projects\creditcardfraud\creditcard.csv')


# In[25]:


data.head()


# In[26]:


data.tail()


# In[27]:


print(data.columns)


# In[28]:


print(data.shape)


# In[29]:


print(data.size)


# In[30]:


print(data.describe())


# In[31]:


data1=data.sample(frac=0.4,random_state=1)


# In[32]:


print(data1.shape)
print(data.shape)


# In[34]:


#plotting the data
data.hist(figsize=(20,20))
plt.show()


# In[33]:


data1.hist(figsize=(20,20))
plt.show()


# In[37]:


#determine number of fraud cases in dataset
Fraud=data[data['Class']==1]
valid=data[data['Class']==0]


# In[38]:


outlier_fraction=len(Fraud)/float(len(valid))
print(outlier_fraction)
print('Fraud cases: {}'.format(len(Fraud)))
print('Valid cases: {}'.format(len(valid)))


# In[39]:


Fraud=data1[data1['Class']==1]
valid=data1[data1['Class']==0]


# In[40]:


outlier_fraction=len(Fraud)/float(len(valid))
print(outlier_fraction)
print('Fraud cases: {}'.format(len(Fraud)))
print('Valid cases: {}'.format(len(valid)))


# In[43]:


#coorelation 
corrmat=data1.corr()
fig=plt.figure(figsize=(12,9))

sns.heatmap(corrmat,vmax=.8,square=True)
plt.show()


# In[64]:


columns=data1.columns.tolist()

#filter the columns to remove data we do not want
columns=[c for c in columns if c not in ['Class']]

#store the variable we ll be predicting on
target="Class"

X=data1[columns]
Y=data1[target]

#print the shapes of X and y
print(X.shape)
print(Y.shape)


# In[49]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# In[51]:


#define a random state

state=1

#define the outlier detection methods

classifiers={"Isolation Forest": IsolationForest(max_samples=len(X),
                                                 contamination=outlier_fraction,random_state=state),
                                                "Local Outlier Factor":LocalOutlierFactor
             (n_neighbors=20,contamination=outlier_fraction)}


# In[68]:


#fit the model
n_outliers=len(Fraud)

for i, (clf_name,clf) in enumerate(classifiers.items()):
    # fit the data and tag outlier
    if clf_name=="Local Outlier Factor":
        y_pred=clf.fit_predict(X)
        scores_pred=clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred=clf.decision_function(X)
        y_pred=clf.predict(X)
    #Reshape the prediction valued to 0 for valid and 1 for fraud
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
    
    n_errors=(y_pred!=Y).sum()
    #run classification metrices
    print('{}: {}'.format(clf_name,n_errors))
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y,y_pred))


# In[ ]:




