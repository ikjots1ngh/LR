#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv(r"C:\Users\Welcome\Desktop\Titanic_train.csv")
df


# In[3]:


df.head(),df.tail()


# In[5]:


df.info(),df.describe,df.shape


# In[7]:


print(type(df))


# In[9]:


z=df.corr(numeric_only=True)
z


# In[11]:


import seaborn as sns
sns.heatmap(z,annot=True)


# In[13]:


df


# In[15]:


df.isnull().sum()


# In[17]:


#drop na values from dataset
df=df.dropna()
df


# In[19]:


df=df.drop("Cabin",axis=1)


# In[21]:


df=df.drop("Embarked",axis=1)


# In[23]:


df=df.drop("Name",axis=1)


# In[25]:


df=df.drop("Sex",axis=1)


# In[27]:


df=df.drop("Ticket",axis=1)


# In[29]:


df=df.drop("PassengerId",axis=1)


# In[31]:


df


# In[33]:


df.shape


# In[35]:


summary=df.describe().T
summary


# In[37]:


#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
df.hist(bins=20,figsize=(15,10),color='skyblue',edgecolor='black')
plt.suptitle("histogram of numerical values",fontsize=16)
plt.show()


# In[39]:


sns.pairplot(df)
plt.show()


# In[41]:


print(type(plt.figure)) 


# In[43]:


df


# In[47]:


X = df.drop('Survived', axis=1)  # Features
y = df['Survived']  # Target variable


# In[49]:


y


# In[53]:


X


# In[55]:


from sklearn.linear_model import LogisticRegression


# In[61]:


#Logistic regression and fit the model
classifier = LogisticRegression()
classifier.fit(X,y)


# In[63]:


#Predict for X dataset
y_pred = classifier.predict(X)
y_pred


# In[65]:


len(y_pred)


# In[67]:


classifier.predict([[1,0,34,12.5,2],[0,1,25,15.6,4]])


# In[69]:


classifier.predict(X)


# In[73]:


y_pred_df= pd.DataFrame({'actual': y,
                         'predicted': classifier.predict(X)})


# In[75]:


y_pred_df


# In[77]:


len(y_pred_df[(y_pred_df["actual"]== 1) & (y_pred_df["predicted"]==1)])


# In[81]:


# Confusion Matrix for the model accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y,y_pred_df["predicted"])
print (confusion_matrix)


# In[83]:


import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(confusion_matrix, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);


# In[85]:


y_pred_df["predicted"].value_counts()


# In[89]:


#Classification report
## Precision – What percent of your predictions were correct?
## Recall – What percent of the positive cases did you catch? 
## F1 score – What percent of positive predictions were correct?
from sklearn.metrics import classification_report
print(classification_report(y,y_pred))


# In[91]:


classifier.predict_proba(X)


# In[93]:


# intercept value
classifier.intercept_


# In[95]:


# Other coeffieicents
classifier.coef_


# In[101]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(y, classifier.predict_proba (X)[:,1])

auc = roc_auc_score(y, y_pred)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='red', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
