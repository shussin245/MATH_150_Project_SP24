#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
import numpy as np


# In[2]:


df = pd.read_csv("data.csv")
df.head()
df


# In[3]:


df['Target'] 


# In[4]:


df.info()


# In[5]:


df.rename(columns = {'Nacionality':'Nationality', 'Age at enrollment':'Age'}, inplace = True)


# In[6]:


df.isnull().sum()/len(df)*100


# In[7]:


df_enrolled = df.loc[df['Target'] == 'Enrolled']
df_enrolled


# In[8]:


df = df[df.Target != "Enrolled"]
df


# In[9]:


print(df["Target"].unique())


# In[10]:


df['Target'] = df['Target'].map({'Graduate':1,'Dropout':0})


# In[11]:


print(df['Target'].unique())


# In[12]:


df.corr()['Target']


# In[13]:


plt.figure(figsize=(30, 30))
sns.heatmap(df.corr() , annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[14]:


drop = df.corr()['Target'] 
drop


# In[15]:


drop = abs(drop.sort_values())
drop = drop.sort_values()
d = drop > 0.05
drop


# In[40]:


d


# In[16]:


new_data = df.copy()
new_data = new_data.drop(columns=["Mother's occupation",'Unemployment rate',"Father's qualification",
                                 "Father's occupation",'International','Educational special needs',
                                 'Nationality','Inflation rate', 'Course', 
                                  'Curricular units 1st sem (credited)'], axis=1)


# In[17]:


new_data.info()


# In[18]:


new_data['Target'].value_counts()


# In[19]:


x = new_data['Target'].value_counts().index
y = new_data['Target'].value_counts().values


# In[20]:


dt = pd.DataFrame({
 'Target': x,
 'Count_T' : y
})
fig = px.pie(dt,
 names ='Target',
 values ='Count_T',
 title='How many dropouts & graduates are there in Target column')
fig.update_traces(labels=['Graduate','Dropout'], hole=0.4,textinfo='value+label', pull=[0,0.2,0.1])
fig.show()


# In[21]:


correlations = df.corr()['Target']
top_10_features = correlations.abs().nlargest(10).index
top_10_corr_values = correlations[top_10_features]
plt.figure(figsize=(10, 11))
plt.bar(top_10_features, top_10_corr_values)
plt.xlabel('Features')
plt.ylabel('Correlation with Target')
plt.title('Top 10 Features with Highest Correlation to Target')
plt.xticks(rotation=45)
plt.show()
top_10_features


# In[22]:


px.histogram(new_data['Age'], x='Age',color_discrete_sequence=['lightblue'])


# In[23]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Age', data=new_data)
plt.xlabel('Target')
plt.ylabel('Age')
plt.title('Relationship between Age and Target')
plt.show()


# In[24]:


X = new_data.drop('Target', axis=1)
y = new_data['Target']


# In[25]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# In[26]:


lr = LogisticRegression(random_state=42)


# In[27]:


lr.fit(X_train,y_train)


# In[28]:


y_pred = lr.predict(X_test)
print("Accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")
print ("AUC Score : ", round(roc_auc_score(y_test,y_pred),2))


# In[29]:


df_enrolled


# In[30]:


x_enrolled = df_enrolled.drop('Target', axis=1)


# In[31]:


x_enrolled


# In[32]:


new_data_enrolled = df_enrolled.copy()
new_data_enrolled = new_data_enrolled.drop(columns=["Mother's occupation",'Unemployment rate',"Father's qualification",
                                 "Father's occupation",'International','Educational special needs',
                                 'Nationality','Inflation rate', 'Course', 
                                  'Curricular units 1st sem (credited)'], axis=1)


# In[33]:


new_data_enrolled


# In[34]:


x_enrolled = new_data_enrolled.drop('Target', axis=1)


# In[35]:


x_enrolled


# In[36]:


y_pred_enrolled = lr.predict(x_enrolled)


# In[37]:


y_pred_enrolled


# In[38]:


unique, counts = np.unique(y_pred_enrolled, return_counts=True)


# In[39]:


dict(zip(unique,counts))

