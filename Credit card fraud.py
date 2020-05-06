
# coding: utf-8

# In[4]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn

print('Python:{}'.format(sys.version))
print('Numpy:{}'.format(numpy.__version__))
print('Pandas:{}'.format(pandas.__version__))
print('Matplotlib:{}'.format(matplotlib.__version__))
print('Seaborn:{}'.format(seaborn.__version__))
print('Scipy:{}'.format(scipy.__version__))
print('Sklearn:{}'.format(sklearn.__version__))


# In[7]:


# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


#Load the dataset from the csv file using pandas
data=pd.read_csv('F:\ML Training\Project\creditcard.csv')


# In[9]:


#explore the dataset
print(data.columns)   


# In[10]:


print(data.shape)


# In[11]:


print(data.describe())


# In[13]:


data=data.sample(frac=0.1, random_state =1)

print(data.shape)


# In[14]:


#Plot histogram of each parameter
data.hist(figsize=(20,20))
plt.show()


# In[15]:


#determine number of fraud cases in dataset
Fraud=data[data['Class']==1]
Valid=data[data['Class']==0]

outlier_fraction=len(Fraud)/float(len(data))
print(outlier_fraction)

print('Fraud Cases:{}'.format(len(Fraud)))
print('Valid Cases:{}'.format(len(Valid)))


# In[16]:


#Correlation matrix
corrmat=data.corr()
fig=plt.figure(figsize=(12,9))

sns.heatmap(corrmat,vmax =.8, square =True)
plt.show()


# In[17]:


# Get all the columns from the Dataframe
columns=data.columns.tolist()

#filter the columns to remove the data we don't want
columns=[c for c in columns if c not in ["Class"]]

#Store the variable we'll be predicting on
target="Class"
X=data[columns]
Y=data[target]

#print the shapes of X and Y
print(X.shape)
print(Y.shape)


# In[20]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#define a random state
state=1

#define tyhe outlier detection methods
classifiers={
    "Isolation Forest":IsolationForest(max_samples=len(X),
                                      contamination=outlier_fraction,
                                      random_state =state),
    "Local Outlier Factor": LocalOutlierFactor(
    n_neighbors = 20,
    contamination =outlier_fraction)
}


# In[26]:


# Fit the model
n_outliers = len(Fraud)

for i,(clf_name,clf) in enumerate(classifiers.items()):
    
    #fit the data and tag outliers
    
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
    #Reshape the prediction values to a 0 for valid, 1 for fraud
    
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    #Run classification metrics
    print('{}:{}'.format(clf_name, n_errors))
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y, y_pred))
     
        

