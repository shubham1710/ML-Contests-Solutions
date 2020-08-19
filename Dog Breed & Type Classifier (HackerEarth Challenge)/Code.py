#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# In[15]:


train = pd.read_csv('train.csv', delimiter=',')
test = pd.read_csv('test.csv', delimiter=',')


# In[16]:


train.head()


# In[17]:


train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(),inplace=True)


# In[20]:


train.head()


# In[26]:


labels = train['color_type'].astype('category').cat.categories.tolist()
replace = {'color_type' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
print(replace)


# In[28]:


train.replace(replace, inplace=True)
train.head()


# In[29]:


labels_test = test['color_type'].astype('category').cat.categories.tolist()
replace_test = {'color_type' : {k: v for k,v in zip(labels,list(range(1,len(labels_test)+1)))}}
print(replace_test)


# In[30]:


test.replace(replace, inplace=True)
test.head()


# In[32]:


train = train.drop(['pet_id', 'issue_date', 'listing_date'], axis=1)
train.head()


# In[34]:


test = test.drop(['pet_id', 'issue_date', 'listing_date'], axis=1)
test.head()


# In[35]:


train_y_breed = train['breed_category']
train_y_pet = train['pet_category']


# In[38]:


train_y_pet.head()


# In[39]:


train_x = train.drop(['breed_category', 'pet_category'], axis=1)
train_x.head()


# In[123]:


x_train_breed,x_val_breed,y_train_breed,y_val_breed=train_test_split(train_x,train_y_breed,test_size=0.2)
x_train_pet,x_val_pet,y_train_pet,y_val_pet=train_test_split(train_x,train_y_pet,test_size=0.2)


# In[124]:


log_breed = RandomForestClassifier(max_depth=15, random_state=45)
log_pet = RandomForestClassifier(max_depth=15, random_state=45)


# In[125]:


log_breed.fit(x_train_breed, y_train_breed)
log_pet.fit(x_train_pet, y_train_pet)


# In[126]:


pred_breed = log_breed.predict(x_val_breed)
pred_pet = log_pet.predict(x_val_pet)


# In[127]:


print(f1_score(y_val_breed, pred_breed, average='weighted'))
print(f1_score(y_val_pet, pred_pet, average='weighted'))


# In[128]:


clf_breed = RandomForestClassifier(max_depth=15, random_state=45)
clf_pet = RandomForestClassifier(max_depth=15, random_state=45)


# In[129]:


clf_breed.fit(train_x, train_y_breed)
clf_pet.fit(train_x, train_y_pet)


# In[130]:


predict_breed = clf_breed.predict(test)
predict_pet = clf_pet.predict(test)


# In[132]:


ro=np.shape(test)[0]


# In[134]:


predict_breed.resize(ro,1)
predict_pet.resize(ro,1)


# In[137]:


test_data = pd.read_csv('test.csv', delimiter=',')


# In[138]:


pet_id = test_data['pet_id']
pet_id.head()


# In[139]:


solution = pd.DataFrame(pet_id, columns =['pet_id'])
solution.head()


# In[140]:


solution['breed_category'] = predict_breed
solution['pet_category'] = predict_pet
solution.head()


# In[142]:


solution["breed_category"] = solution["breed_category"].astype(int)
solution.head()


# In[144]:


solution.to_csv('prediction.csv',index=False)

