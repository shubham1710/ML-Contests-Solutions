import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('train.csv', delimiter=',')
test = pd.read_csv('test.csv', delimiter=',')

train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(),inplace=True)

labels = train['color_type'].astype('category').cat.categories.tolist()
replace = {'color_type' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}

train.replace(replace, inplace=True)

labels_test = test['color_type'].astype('category').cat.categories.tolist()
replace_test = {'color_type' : {k: v for k,v in zip(labels,list(range(1,len(labels_test)+1)))}}

test.replace(replace, inplace=True)

train = train.drop(['pet_id', 'issue_date', 'listing_date'], axis=1)

test = test.drop(['pet_id', 'issue_date', 'listing_date'], axis=1)

train_y_breed = train['breed_category']
train_y_pet = train['pet_category']

train_x = train.drop(['breed_category', 'pet_category'], axis=1)

x_train_breed,x_val_breed,y_train_breed,y_val_breed=train_test_split(train_x,train_y_breed,test_size=0.2)
x_train_pet,x_val_pet,y_train_pet,y_val_pet=train_test_split(train_x,train_y_pet,test_size=0.2)

log_breed = RandomForestClassifier(max_depth=15, random_state=45)
log_pet = RandomForestClassifier(max_depth=15, random_state=45)

log_breed.fit(x_train_breed, y_train_breed)
log_pet.fit(x_train_pet, y_train_pet)

pred_breed = log_breed.predict(x_val_breed)
pred_pet = log_pet.predict(x_val_pet)

print(f1_score(y_val_breed, pred_breed, average='weighted'))
print(f1_score(y_val_pet, pred_pet, average='weighted'))

clf_breed = RandomForestClassifier(max_depth=15, random_state=45)
clf_pet = RandomForestClassifier(max_depth=15, random_state=45)

clf_breed.fit(train_x, train_y_breed)
clf_pet.fit(train_x, train_y_pet)

predict_breed = clf_breed.predict(test)
predict_pet = clf_pet.predict(test)

ro=np.shape(test)[0]

predict_breed.resize(ro,1)
predict_pet.resize(ro,1)

test_data = pd.read_csv('test.csv', delimiter=',')

pet_id = test_data['pet_id']

solution = pd.DataFrame(pet_id, columns =['pet_id'])

solution['breed_category'] = predict_breed
solution['pet_category'] = predict_pet

solution["breed_category"] = solution["breed_category"].astype(int)

solution.to_csv('prediction.csv',index=False)

