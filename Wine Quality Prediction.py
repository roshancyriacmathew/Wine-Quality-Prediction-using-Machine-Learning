#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[2]:


wine_df = pd.read_csv('winequality-red.csv', sep=';')
wine_df.head()


# ## EDA 

# In[4]:


wine_df.shape


# In[5]:


wine_df.info()


# In[6]:


wine_df.isnull().sum()


# In[7]:


wine_df.describe()


# In[8]:


wine_df['quality'].value_counts()


# In[9]:


style.use('ggplot')
sns.countplot(wine_df['quality'])


# In[10]:


wine_df.hist(bins=100, figsize=(10,12))
plt.show()


# In[11]:


plt.figure(figsize=(10,7))
sns.heatmap(wine_df.corr(), annot=True)
plt.title('Correlation between the columns')
plt.show()


# In[13]:


wine_df.corr()['quality'].sort_values()


# In[14]:


sns.barplot(wine_df['quality'], wine_df['alcohol'])


# ## Data Processing 

# In[16]:


wine_df['quality'] = wine_df.quality.apply(lambda x:1 if x>=7 else 0)


# In[17]:


wine_df['quality'].value_counts()


# In[18]:


X = wine_df.drop('quality', axis=1)
y = wine_df['quality']


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# In[22]:


print("X_train ", X_train.shape)
print("y_train ", y_train.shape)
print("X_test ", X_test.shape)
print("y_test ", y_test.shape)


# ## Model Training 

# #### logistic Regression model

# In[23]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)
logreg_acc = accuracy_score(logreg_pred, y_test)
print("test accuracy is: {:.2f}%".format(logreg_acc*100))


# In[24]:


print(classification_report(y_test, logreg_pred))


# In[25]:


style.use('classic')
cm = confusion_matrix(y_test, logreg_pred, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix= cm, display_labels=logreg.classes_)
disp.plot()
print("TN: ", cm[0][0])
print("FN: ", cm[1][0])
print("TP: ", cm[1][1])
print("FP: ", cm[0][1])


# #### Decision Tree 

# In[26]:


dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
dtree_pred = dtree.predict(X_test)
dtree_acc = accuracy_score(dtree_pred, y_test)
print("Test accuracy: {:.2f}%".format(dtree_acc*100))


# In[27]:


print(classification_report(y_test, dtree_pred))


# In[28]:


style.use('classic')
cm = confusion_matrix(y_test, dtree_pred, labels=dtree.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix= cm, display_labels=dtree.classes_)
disp.plot()
print("TN: ", cm[0][0])
print("FN: ", cm[1][0])
print("TP: ", cm[1][1])
print("FP: ", cm[0][1])


# #### Random Forest  

# In[29]:


rforest = RandomForestClassifier()
rforest.fit(X_train, y_train)
rforest_pred = rforest.predict(X_test)
rforest_acc = accuracy_score(rforest_pred, y_test)
print("Test accuracy: {:.2f}%".format(rforest_acc*100))


# In[30]:


print(classification_report(y_test, rforest_pred))


# In[31]:


style.use('classic')
cm = confusion_matrix(y_test, rforest_pred, labels=rforest.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix= cm, display_labels=rforest.classes_)
disp.plot()
print("TN: ", cm[0][0])
print("FN: ", cm[1][0])
print("TP: ", cm[1][1])
print("FP: ", cm[0][1])

