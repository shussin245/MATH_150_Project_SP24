#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.linear_model import LogisticRegression


# In[4]:


df = pd.read_csv("data.csv")
df.head()
df


# In[5]:


df['Target'] 


# In[6]:


df.info()


# In[7]:


df.rename(columns = {'Nacionality':'Nationality', 'Age at enrollment':'Age'}, inplace = True)


# In[8]:


df.isnull().sum()/len(df)*100


# In[9]:


df_enrolled = df.loc[df['Target'] == 'Enrolled']
df_enrolled


# In[10]:


df = df[df.Target != "Enrolled"]
df


# In[11]:


print(df["Target"].unique())


# In[12]:


df['Target'] = df['Target'].map({'Graduate':1,'Dropout':0})


# In[13]:


print(df['Target'].unique())


# In[14]:


df.corr()['Target']


# In[15]:


plt.figure(figsize=(30, 30))
sns.heatmap(df.corr() , annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[16]:


drop = df.corr()['Target'] 


# In[17]:


drop = abs(drop.sort_values())
drop = drop.sort_values()
d = drop > 0.05
d


# In[18]:


new_data = df.copy()
new_data = new_data.drop(columns=["Mother's occupation",'Unemployment rate',"Father's qualification",
                                 "Father's occupation",'International','Educational special needs',
                                 'Nationality','Inflation rate', 'Course', 
                                  'Curricular units 1st sem (credited)'], axis=1)


# In[19]:


new_data.info()


# In[20]:


new_data['Target'].value_counts()


# In[21]:


x = new_data['Target'].value_counts().index
y = new_data['Target'].value_counts().values


# In[22]:


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


# In[23]:


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


# In[24]:


px.histogram(new_data['Age'], x='Age',color_discrete_sequence=['lightblue'])


# In[25]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Age', data=new_data)
plt.xlabel('Target')
plt.ylabel('Age')
plt.title('Relationship between Age and Target')
plt.show()


# In[26]:


X = new_data.drop('Target', axis=1)
y = new_data['Target']


# In[27]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# In[28]:


lr = LogisticRegression(random_state=42)


# In[29]:


lr.fit(X_train,y_train)


# In[30]:


y_pred = lr.predict(X_test)
print("Accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")


# In[ ]:





# In[ ]:




