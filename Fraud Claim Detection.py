#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt

import seaborn as sns


# In[3]:


pwd


# In[4]:


cd"G:\Travelers\train_2021"


# In[5]:


train_data  = pd.read_csv("train_2021.csv")


# In[6]:


pd.set_option('display.max_columns',30)


# In[7]:


train_data.head()


# In[8]:


train_data.shape


# In[9]:


train_data.describe()


# In[10]:


g = sns.countplot(train_data['fraud'])
g.set_xticklabels(['Not Fraud','Fraud'])
plt.show()


# In[11]:


# A widely adopted technique for dealing with highly unbalanced datasets is called resampling. 
# It consists of removing samples from the majority class (under-sampling)
# and/or adding more examples from the minority class (over-sampling).


# In[12]:


# class count
class_count_0, class_count_1 = train_data['fraud'].value_counts()

# Separate class
class_0 = train_data[train_data['fraud'] == 0]
class_1 = train_data[train_data['fraud'] == 1]# print the shape of the class
print('class 0:', class_0.shape)
print('class 1:', class_1.shape)


# In[13]:


class_0_under = class_0.sample(class_count_1)

test_under = pd.concat([class_0_under, class_1], axis=0)

print("total class of 1 and0:",test_under['fraud'].value_counts())# plot the count after under-sampeling
test_under['fraud'].value_counts().plot(kind='bar', title='count (target)')


# In[14]:


class_1_over = class_1.sample(class_count_0, replace=True)

test_over = pd.concat([class_1_over, class_0], axis=0)

print("total class of 1 and 0:",test_under['fraud'].value_counts())# plot the count after under-sampeling
test_over['fraud'].value_counts().plot(kind='bar', title='count (target)')


# In[15]:


test_over.shape


# In[16]:


test_over.columns


# In[17]:


test_over.isna().sum()


# In[18]:


train_data=test_under.copy(deep=True)


# In[19]:


train_data.isna().sum()


# In[20]:


train_data["marital_status"].median()


# In[21]:


marital_status_median = train_data["marital_status"].median()


# In[ ]:





# In[22]:


train_data['marital_status'] = train_data['marital_status'].fillna(marital_status_median)


# In[23]:




claim_est_payout_mean = train_data["claim_est_payout"].mean()
train_data['claim_est_payout'] = train_data['claim_est_payout'].fillna(claim_est_payout_mean)

age_of_vehicle_mean = train_data["age_of_vehicle"].mean()
train_data['age_of_vehicle'] = train_data['age_of_vehicle'].fillna(age_of_vehicle_mean)


# In[24]:


witness_median = train_data["witness_present_ind"].median()
train_data['witness_present_ind'] = train_data['witness_present_ind'].fillna(witness_median)


# In[25]:


train_data.isna().sum()


# In[26]:


train_data["claim_number"].nunique()


# In[27]:


train_data.shape


# In[28]:


train_data["age_of_driver"].median()


# In[29]:


sliced_data = train_data[train_data["age_of_driver"]<=95]
wrong_age_data = train_data[train_data["age_of_driver"]>95]
wrong_age_data["age_of_driver"]=sliced_data["age_of_driver"].median()
age_corrected_train_data = pd.concat([sliced_data, wrong_age_data], ignore_index=True, sort=False)


# In[30]:


age_corrected_train_data.shape


# In[31]:


age_corrected_train_data["age_of_driver"].max()


# In[32]:


age_corrected_train_data.head()


# In[ ]:





# In[33]:


income_sliced_data = age_corrected_train_data[age_corrected_train_data["annual_income"]>0]
wrong_income_data = age_corrected_train_data[age_corrected_train_data["annual_income"]<=0]
wrong_income_data["annual_income"]=income_sliced_data["annual_income"].mean()
income_corrected_train_data = pd.concat([income_sliced_data, wrong_income_data], ignore_index=True, sort=False)


# In[34]:


income_corrected_train_data.shape


# In[35]:


income_corrected_train_data["annual_income"].min()


# In[36]:


income_corrected_train_data.dtypes


# In[ ]:





# In[37]:


income_corrected_train_data=income_corrected_train_data.astype({"age_of_driver" : int})
income_corrected_train_data=income_corrected_train_data.astype({"marital_status" : int})
income_corrected_train_data=income_corrected_train_data.astype({"zip_code" : str})
income_corrected_train_data=income_corrected_train_data.astype({"witness_present_ind" : int})
income_corrected_train_data=income_corrected_train_data.astype({"age_of_vehicle" : int})


# In[ ]:





# In[38]:


income_corrected_train_data['claim_date'] = pd.to_datetime(income_corrected_train_data['claim_date'], format='%m/%d/%Y')


# In[39]:


income_corrected_train_data["zip_code"].nunique()


# In[40]:


income_corrected_train_data.head()


# In[41]:


income_corrected_train_data['Month']=income_corrected_train_data['claim_date'].dt.month
income_corrected_train_data['Year']=income_corrected_train_data['claim_date'].dt.year


# In[42]:


income_corrected_train_data=income_corrected_train_data.astype({"Month" : str})
income_corrected_train_data=income_corrected_train_data.astype({"Year" : str})


# In[43]:


from sklearn import preprocessing


# encode categorical variables using Label Encoder

# select all categorical variables
df_categorical = income_corrected_train_data.select_dtypes(include=['object'])



# In[44]:


income_corrected_train_data.dtypes


# In[45]:


df_categorical.head()


# In[46]:


# apply Label encoder to df_categorical

le = preprocessing.LabelEncoder()
df_categorical = df_categorical.apply(le.fit_transform)


# In[47]:



# concat df_categorical with original df
encoded_train_data = income_corrected_train_data.drop(df_categorical.columns, axis=1)
encoded_train_data = pd.concat([encoded_train_data, df_categorical], axis=1)


# In[48]:


encoded_train_data.head()


# In[49]:


encoded_train_data.corr(method='pearson', min_periods=1)


# In[50]:


# The target variable is less correlated with LIABILITY_PERCENTAGE,VEHICLE_PRICE,CHANNEL,VEHICLE_CATEGORY,VEHICLE_COLOR
# So, We have decided to drop these 


# In[ ]:





# In[51]:


encoded_train_data.head()


# In[52]:


c = encoded_train_data.corr().abs()
s = c.unstack()
so = s.sort_values(kind="quicksort")


# In[53]:


pd.set_option('display.max_rows',30)


# In[54]:


so>0.9


# In[55]:


upper_tri = c.where(np.triu(np.ones(c.shape),k=1).astype(np.bool))


# In[56]:


to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
print(to_drop)


# In[57]:


# Annual_income is highly correlated with age_of_driver


# In[58]:


after_cor_dropped = encoded_train_data.drop(['annual_income'],axis=1)


# In[59]:


after_cor_dropped.head()


# In[60]:


after_cor_dropped['log_claim_est_payout'] = np.log2(after_cor_dropped['claim_est_payout'])


# In[61]:



after_cor_dropped['log_vehicle_price'] = np.log2(after_cor_dropped['vehicle_price'])
after_cor_dropped['log_vehicle_weight'] = np.log2(after_cor_dropped['vehicle_weight'])


# In[62]:


after_cor_dropped_transformed = after_cor_dropped.drop(['claim_est_payout','vehicle_price','vehicle_weight'],axis=1)


# In[63]:


after_cor_dropped_transformed.head()


# In[64]:


after_cor_dropped_transformed = after_cor_dropped_transformed.drop(['claim_date'],axis=1)


# In[65]:


after_cor_dropped_transformed['Year'].unique()


# In[66]:


after_cor_dropped_transformed = after_cor_dropped_transformed.drop(['claim_number'],axis=1)


# In[ ]:





# In[67]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[68]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


# In[69]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)


# In[70]:


test_data  = pd.read_csv("test_2021.csv")


# In[71]:


test_data.head()


# In[72]:


test_data.isna().sum()


# In[73]:


marital_status_median = test_data["marital_status"].median()
test_data['marital_status'] = test_data['marital_status'].fillna(marital_status_median)
age_of_vehicle_mean = test_data["age_of_vehicle"].mean()
test_data['age_of_vehicle'] = test_data['age_of_vehicle'].fillna(age_of_vehicle_mean)

witness_median = test_data["witness_present_ind"].median()
test_data['witness_present_ind'] = test_data['witness_present_ind'].fillna(witness_median)


# In[ ]:





# In[74]:


test_data=test_data.astype({"age_of_driver" : int})
test_data=test_data.astype({"marital_status" : int})
test_data=test_data.astype({"zip_code" : str})
test_data=test_data.astype({"witness_present_ind" : int})
test_data=test_data.astype({"age_of_vehicle" : int})


# In[75]:


test_data['claim_date'] = pd.to_datetime(test_data['claim_date'], format='%m/%d/%Y')
test_data['Month']=test_data['claim_date'].dt.month
test_data['Year']=test_data['claim_date'].dt.year
test_data=test_data.astype({"Month" : str})
test_data=test_data.astype({"Year" : str})


# In[76]:


test_data = test_data.drop(['claim_date'],axis=1)


# In[77]:





# encode categorical variables using Label Encoder

# select all categorical variables
df_categorical = test_data.select_dtypes(include=['object'])



# In[78]:


# apply Label encoder to df_categorical

le = preprocessing.LabelEncoder()
df_categorical = df_categorical.apply(le.fit_transform)


# In[79]:



# concat df_categorical with original df
encoded_test_data = test_data.drop(df_categorical.columns, axis=1)
encoded_test_data = pd.concat([encoded_test_data, df_categorical], axis=1)


# In[80]:


encoded_test_data.shape


# In[81]:


encoded_test_data['log_claim_est_payout'] = np.log2(encoded_test_data['claim_est_payout'])


# In[82]:



encoded_test_data['log_vehicle_price'] = np.log2(encoded_test_data['vehicle_price'])
encoded_test_data['log_vehicle_weight'] = np.log2(encoded_test_data['vehicle_weight'])
encoded_test_data = encoded_test_data.drop(['claim_est_payout','vehicle_price','vehicle_weight'],axis=1)


# In[83]:


encoded_test_data.isna().sum()


# In[84]:


log_claim_est_payout_mean = encoded_test_data["log_claim_est_payout"].mean()


# In[85]:


encoded_test_data['log_claim_est_payout'] = encoded_test_data['log_claim_est_payout'].fillna(log_claim_est_payout_mean)


# In[86]:


after_cor_dropped_transformed.columns


# In[87]:


encoded_test_data = encoded_test_data.drop(['claim_number'],axis=1)


# In[88]:


encoded_test_data['fraud']=0


# In[89]:


encoded_test_data.head()


# In[90]:


X_train = after_cor_dropped_transformed.drop(['fraud'],axis=1)
Y_train = after_cor_dropped_transformed['fraud']
X_test = encoded_test_data.drop(['fraud'],axis=1)
Y_test = encoded_test_data['fraud']


# In[91]:


#rf_random.fit(X_train,Y_train)


# In[92]:


#rf_random.best_params_


# In[93]:


model = RandomForestClassifier(n_estimators= 1000,
 min_samples_split= 2,
 min_samples_leaf= 1,
 max_features='auto',
 max_depth= 50,
 bootstrap=False)


# In[94]:


model_2 = RandomForestClassifier(n_estimators= 1800,
 min_samples_split= 2,
 min_samples_leaf= 4,
 max_features='sqrt',
 max_depth= 90,
 bootstrap=True)


# In[95]:


model_2.fit(X_train,Y_train)


# In[96]:


X_test.isna().sum()


# In[97]:


X_train.shape


# In[98]:


X_train.columns


# In[99]:


X_test = X_test.drop(['annual_income'],axis=1)


# In[100]:


X_test.columns


# In[101]:


X_test.shape


# In[102]:


predictions = model_2.predict(X_test)


# In[103]:


predictions_df = pd.DataFrame(predictions, columns=['fraud'])


# In[ ]:





# In[104]:


predictions_df["fraud"].value_counts()


# In[105]:


test_data_1  = pd.read_csv("test_2021.csv")


# In[106]:


df_concat = pd.concat([test_data_1[["claim_number"]],predictions_df], axis=1)


# In[107]:


df_concat.head()


# In[108]:


df_concat['fraud'].value_counts()


# In[109]:


df_concat.to_csv("submission_2.csv",index=False)


# In[110]:


df_concat.to_csv("submission.csv",index=False)


# In[111]:


import time


start_time = time.time()
importances = model_2.feature_importances_
std = np.std([tree.feature_importances_ for tree in model_2.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")


# In[112]:


after_cor_dropped_transformed.shape


# In[113]:


X_train.columns


# In[114]:


feature_names = [f"feature {i}" for i in range(X_train.shape[1])]

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


# In[115]:


# We can remove the features that are less important in building our tree
# Features to be removed are as follows
# Feature 1: marital_status
# Feature 4: address_change_ind
# Feature 8: policy_report_filed_ind
# Feature 10: gender
# Feature 11: living_status
# Feature 19: Year
# Feature 3: high_education_ind


# Feature 15: channel


# In[116]:


after_cor_dropped_transformed.drop(['marital_status','address_change_ind','policy_report_filed_ind','gender','living_status','Year','channel'],axis=1,inplace=True)


# In[117]:



encoded_test_data.drop(['marital_status','address_change_ind','policy_report_filed_ind','gender','living_status','Year','channel'],axis=1,inplace=True)


# In[ ]:





# In[118]:


X_train = after_cor_dropped_transformed.drop(['fraud'],axis=1)
Y_train = after_cor_dropped_transformed['fraud']
X_test = encoded_test_data.drop(['fraud'],axis=1)
Y_test = encoded_test_data['fraud']


# In[119]:


model_2.fit(X_train,Y_train)


# In[120]:


X_test = X_test.drop(['annual_income'],axis=1)


# In[121]:


predictions = model_2.predict(X_test)


# In[122]:


predictions_df = pd.DataFrame(predictions, columns=['pred'])


# In[123]:


predictions_df["pred"].value_counts()


# In[124]:


df_concat = pd.concat([test_data_1[["claim_number"]],predictions_df], axis=1)


# In[125]:


df_concat.head()


# In[126]:


df_concat.to_csv("submission_1.csv",index=False)


# In[ ]:




