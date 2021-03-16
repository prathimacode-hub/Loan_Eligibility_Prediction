#!/usr/bin/env python
# coding: utf-8

# In[135]:


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# In[136]:


train=pd.read_csv(r'C:\Users\prath\LoanEligibilityPrediction\Dataset\train.csv')
train.Loan_Status=train.Loan_Status.map({'Y':1,'N':0})
train.isnull().sum()


# In[137]:


Loan_status=train.Loan_Status
train.drop('Loan_Status',axis=1,inplace=True)
test=pd.read_csv(r'C:\Users\prath\LoanEligibilityPrediction\Dataset\test.csv')
Loan_ID=test.Loan_ID
data=train.append(test)
data.head()


# In[138]:


data.describe()


# In[139]:


data.isnull().sum()


# In[140]:


data.Dependents.dtypes


# In[141]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[142]:


data.Gender=data.Gender.map({'Male':1,'Female':0})
data.Gender.value_counts()


# In[143]:


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[144]:


data.Married=data.Married.map({'Yes':1,'No':0})


# In[145]:


data.Married.value_counts()


# In[146]:


data.Dependents=data.Dependents.map({'0':0,'1':1,'2':2,'3+':3})


# In[147]:


data.Dependents.value_counts()


# In[148]:


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[149]:


data.Education=data.Education.map({'Graduate':1,'Not Graduate':0})


# In[150]:


data.Education.value_counts()


# In[151]:


data.Self_Employed=data.Self_Employed.map({'Yes':1,'No':0})


# In[152]:


data.Self_Employed.value_counts()


# In[153]:


data.Property_Area.value_counts()


# In[154]:


data.Property_Area=data.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})


# In[155]:


data.Property_Area.value_counts()


# In[156]:


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[157]:


data.head()


# In[158]:


data.Credit_History.size


# In[159]:


data.Credit_History.fillna(np.random.randint(0,2),inplace=True)


# In[160]:


data.isnull().sum()


# In[161]:


data.Married.fillna(np.random.randint(0,2),inplace=True)


# In[162]:


data.isnull().sum()


# In[163]:


data.LoanAmount.fillna(data.LoanAmount.median(),inplace=True)


# In[164]:


data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mean(),inplace=True)


# In[165]:


data.isnull().sum()


# In[166]:


data.Gender.value_counts()


# In[167]:


from random import randint 
data.Gender.fillna(np.random.randint(0,2),inplace=True)


# In[168]:


data.Gender.value_counts()


# In[169]:


data.Dependents.fillna(data.Dependents.median(),inplace=True)


# In[170]:


data.isnull().sum()


# In[171]:


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[172]:


data.Self_Employed.fillna(np.random.randint(0,2),inplace=True)


# In[173]:


data.isnull().sum()


# In[174]:


data.head()


# In[175]:


data.drop('Loan_ID',inplace=True,axis=1)


# In[176]:


data.isnull().sum()


# In[177]:


train_X=data.iloc[:614,]
train_y=Loan_status
X_test=data.iloc[614:,]
seed=7


# In[178]:


from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(train_X,train_y,random_state=seed)


# In[179]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[180]:


models=[]
models.append(("logreg",LogisticRegression()))
models.append(("tree",DecisionTreeClassifier()))
models.append(("lda",LinearDiscriminantAnalysis()))
models.append(("svc",SVC()))
models.append(("knn",KNeighborsClassifier()))
models.append(("nb",GaussianNB()))


# In[181]:


seed=7
scoring='accuracy'


# In[182]:


from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
result=[]
names=[]


# In[203]:


for name,model in models:
    #print(model)
    kfold=KFold(n_splits=10,random_state=seed)
    cv_result=cross_val_score(model,train_X,train_y,cv=kfold,scoring=scoring)
    result.append(cv_result)
    names.append(name)
    print("%s %f %f" % (name,cv_result.mean(),cv_result.std()))


# In[204]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
svc=LogisticRegression()
svc.fit(train_X,train_y)
pred=svc.predict(test_X)
print(accuracy_score(test_y,pred))
print(confusion_matrix(test_y,pred))
print(classification_report(test_y,pred))


# In[205]:


df_output=pd.DataFrame()


# In[206]:


outp=svc.predict(X_test).astype(int)
outp


# In[207]:


df_output['Loan_ID']=Loan_ID
df_output['Loan_Status']=outp


# In[208]:


df_output.head()


# In[209]:


df_output[['Loan_ID','Loan_Status']].to_csv(r'C:\Users\prath\LoanEligibilityPrediction\Dataset\outputlr.csv',index=False)


# In[210]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
svc=DecisionTreeClassifier()
svc.fit(train_X,train_y)
pred=svc.predict(test_X)
print(accuracy_score(test_y,pred))
print(confusion_matrix(test_y,pred))
print(classification_report(test_y,pred))


# In[211]:


df_output=pd.DataFrame()


# In[212]:


outp=svc.predict(X_test).astype(int)
outp


# In[213]:


df_output['Loan_ID']=Loan_ID
df_output['Loan_Status']=outp


# In[214]:


df_output.head()


# In[215]:


df_output[['Loan_ID','Loan_Status']].to_csv(r'C:\Users\prath\LoanEligibilityPrediction\Dataset\outputdt.csv',index=False)


# In[216]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
svc=LinearDiscriminantAnalysis()
svc.fit(train_X,train_y)
pred=svc.predict(test_X)
print(accuracy_score(test_y,pred))
print(confusion_matrix(test_y,pred))
print(classification_report(test_y,pred))


# In[217]:


df_output=pd.DataFrame()


# In[218]:


outp=svc.predict(X_test).astype(int)
outp


# In[219]:


df_output['Loan_ID']=Loan_ID
df_output['Loan_Status']=outp


# In[220]:


df_output.head()


# In[221]:


df_output[['Loan_ID','Loan_Status']].to_csv(r'C:\Users\prath\LoanEligibilityPrediction\Dataset\outputld.csv',index=False)


# In[222]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
svc=SVC()
svc.fit(train_X,train_y)
pred=svc.predict(test_X)
print(accuracy_score(test_y,pred))
print(confusion_matrix(test_y,pred))
print(classification_report(test_y,pred))


# In[223]:


df_output=pd.DataFrame()


# In[224]:


outp=svc.predict(X_test).astype(int)
outp


# In[225]:


df_output['Loan_ID']=Loan_ID
df_output['Loan_Status']=outp


# In[226]:


df_output.head()


# In[227]:


df_output[['Loan_ID','Loan_Status']].to_csv(r'C:\Users\prath\LoanEligibilityPrediction\Dataset\outputSVC.csv',index=False)


# In[228]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
svc=KNeighborsClassifier()
svc.fit(train_X,train_y)
pred=svc.predict(test_X)
print(accuracy_score(test_y,pred))
print(confusion_matrix(test_y,pred))
print(classification_report(test_y,pred))


# In[229]:


df_output=pd.DataFrame()


# In[230]:


outp=svc.predict(X_test).astype(int)
outp


# In[231]:


df_output['Loan_ID']=Loan_ID
df_output['Loan_Status']=outp


# In[232]:


df_output.head()


# In[233]:


df_output[['Loan_ID','Loan_Status']].to_csv(r'C:\Users\prath\LoanEligibilityPrediction\Dataset\outputknn.csv',index=False)


# In[234]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
svc=GaussianNB()
svc.fit(train_X,train_y)
pred=svc.predict(test_X)
print(accuracy_score(test_y,pred))
print(confusion_matrix(test_y,pred))
print(classification_report(test_y,pred))


# In[236]:


df_output=pd.DataFrame()


# In[237]:


outp=svc.predict(X_test).astype(int)
outp


# In[238]:


df_output['Loan_ID']=Loan_ID
df_output['Loan_Status']=outp


# In[239]:


df_output.head()


# In[240]:


df_output[['Loan_ID','Loan_Status']].to_csv(r'C:\Users\prath\LoanEligibilityPrediction\Dataset\outputgnb.csv',index=False)


# In[ ]:




