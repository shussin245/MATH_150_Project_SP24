#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier


# In[2]:


df = pd.read_csv("data.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.rename(columns = {'Nacionality':'Nationality', 'Age at enrollment':'Age'}, inplace = True)


# In[5]:


df.isnull().sum()/len(df)*100


# In[6]:


print(df["Target"].unique())


# In[7]:


df['Target'] = df['Target'].map({
 'Dropout':0,
 'Enrolled':1,
 'Graduate':2
})


# In[8]:


print(df['Target'].unique())


# In[9]:


df.corr()['Target']


# In[10]:


plt.figure(figsize=(30, 30))
sns.heatmap(df.corr() , annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[11]:


drop = df.corr()['Target'] < abs(0.015)


# In[12]:


drop


# In[13]:


new_data = df.copy()
new_data = new_data.drop(columns=['Nationality',
 'Mother\'s qualification',
 'Father\'s qualification',
 'Educational special needs',
'International',
'Curricular units 1st sem (without evaluations)',
 'Unemployment rate',
'Inflation rate'], axis=1)


# In[14]:


new_data.info()


# In[15]:


new_data['Target'].value_counts()


# In[16]:


x = new_data['Target'].value_counts().index
y = new_data['Target'].value_counts().values


# In[17]:


dt = pd.DataFrame({
 'Target': x,
 'Count_T' : y
})
fig = px.pie(dt,
 names ='Target',
 values ='Count_T',
 title='How many dropouts, enrolled & graduates are there in Target column')
fig.update_traces(labels=['Graduate','Dropout','Enrolled'], hole=0.4,textinfo='value+label', pull=[0,0.2,0.1])
fig.show()


# In[18]:


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


# In[19]:


px.histogram(new_data['Age'], x='Age',color_discrete_sequence=['lightblue'])


# In[20]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Age', data=new_data)
plt.xlabel('Target')
plt.ylabel('Age')
plt.title('Relationship between Age and Target')
plt.show()


# In[21]:


X = new_data.drop('Target', axis=1)
y = new_data['Target']


# In[22]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[23]:


dtree = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=2)
lr = LogisticRegression(random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
abc = AdaBoostClassifier(n_estimators=50,learning_rate=1, random_state=
0)
xbc = XGBClassifier(tree_method='gpu_hist')
svm = svm.SVC(kernel='linear',probability=True)


# In[24]:


dtree.fit(X_train,y_train)
rfc.fit(X_train,y_train)
lr.fit(X_train,y_train)
knn.fit(X_train,y_train)
abc.fit(X_train, y_train)
#xbc.fit(X_train, y_train)
svm.fit(X_train, y_train)


# In[25]:


y_pred = dtree.predict(X_test)
print("Accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")


# In[26]:


y_pred = rfc.predict(X_test)
print("Accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")


# In[27]:


y_pred = lr.predict(X_test)
print("Accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")


# In[28]:


y_pred = knn.predict(X_test)
print("Accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")


# In[29]:


y_pred = abc.predict(X_test)
print("Accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")


# In[30]:


#y_pred = xbc.predict(X_test)
#print("Accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")


# In[31]:


y_pred = svm.predict(X_test)
print("Accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")


# In[ ]:




