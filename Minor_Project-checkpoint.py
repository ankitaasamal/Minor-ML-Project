#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


# # DATA LOADING ( SCIKIT LEARN )

# In[2]:


from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()


# In[3]:


cancer_dataset


# # DATA MANIPULATION

# In[4]:


type(cancer_dataset)


# In[5]:


cancer_dataset.keys()


# In[6]:


#features of cells
cancer_dataset['data']


# In[7]:


type(cancer_dataset['data'])


# In[8]:


cancer_dataset['target']


# In[9]:


#benign or malignant tumor
cancer_dataset['target_names']


# In[10]:


print(cancer_dataset['DESCR'])


# In[ ]:





# In[11]:


print(cancer_dataset['feature_names'])
#independent variables


# # DATAFRAME

# In[12]:


df_cancer = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],
             columns = np.append(cancer_dataset['feature_names'], ['target']))


# In[13]:


# DataFrame into CSV file
df_cancer.to_csv('breast_cancer_dataframe.csv')


# In[14]:


#getting the first 10 rows
df_cancer.head(10)


# In[15]:


#getting last 10 rows
df_cancer.tail(10)


# In[16]:


#dataframe description
df_cancer.info()


# In[17]:


#checking for missing values
df_cancer.isnull().sum()


# In[18]:


#numerical distribution
df_cancer.describe()


# # DATA VISUALISATION

# In[ ]:


# Paiplot of cancer dataframe
sns.pairplot(df_cancer, hue = 'target') 


# In[20]:


sns.pairplot(df_cancer, hue = 'target', 
             vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'] )


# In[21]:


# Count the target class
sns.countplot(df_cancer['target'])


# In[22]:


# counter plot of feature mean radius
plt.figure(figsize = (15,7))
sns.countplot(df_cancer['mean radius'])


# # HEATMAP

# In[23]:


# heatmap of DataFrame
plt.figure(figsize=(15,7))
sns.heatmap(df_cancer)


# In[24]:


df_cancer.corr()


# In[25]:


# Heatmap of Correlation matrix of breast cancer DataFrame
plt.figure(figsize=(15,15))
sns.heatmap(df_cancer.corr(), annot = True, cmap ='rainbow', linewidths=2)


# CORELATION BARPLOT

# In[26]:


# create second DataFrame by droping target
df_cancer2 = df_cancer.drop(['target'], axis = 1)
print("The shape of 'cancer_df2' is : ", df_cancer2.shape)


# In[27]:


df_cancer2.corrwith(df_cancer.target)


# In[28]:


#corealtion barplot
plt.figure(figsize = (16,5))
ax = sns.barplot(df_cancer2.corrwith(df_cancer.target).index, df_cancer2.corrwith(df_cancer.target))
ax.tick_params(labelrotation = 90)


# In[29]:


df_cancer2.corrwith(df_cancer.target).index


# # Splitting dataframe 

# In[30]:


# input variable
X = df_cancer.drop(['target'], axis = 1) 
X.head(10)


# In[31]:


# output variable
y = df_cancer['target'] 
y.head(10)


# In[32]:


# split dataset into train and test X features y labels
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state= 5)


# In[33]:


X_train


# In[34]:


X_test


# In[35]:


y_train


# In[36]:


y_test


# # Feature scaling

# In[37]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


# # ML model build

# In[38]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# # Logistic Regression

# In[39]:


from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state = 51, penalty = 'l2')
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)
accuracy_score(y_test, y_pred_lr)


# In[40]:


# Train with Standard scaled Data
lr_classifier2 = LogisticRegression(random_state = 51, penalty = 'l2')
lr_classifier2.fit(X_train_sc, y_train)
y_pred_lr_sc = lr_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_lr_sc)


# # Suppor vector Classifier

# In[41]:


from sklearn.svm import SVC
svc_classifier = SVC()
svc_classifier.fit(X_train, y_train)
y_pred_scv = svc_classifier.predict(X_test)
accuracy_score(y_test, y_pred_scv)


# In[42]:


# Train with Standard scaled Data
svc_classifier2 = SVC()
svc_classifier2.fit(X_train_sc, y_train)
y_pred_svc_sc = svc_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_svc_sc)


# # KNN Classifier

# In[43]:


from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_score(y_test, y_pred_knn)


# In[44]:


# Train with Standard scaled Data
knn_classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier2.fit(X_train_sc, y_train)
y_pred_knn_sc = knn_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_knn_sc)


# # Naive Bayes

# In[45]:


from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_nb)


# In[46]:


# Train with Standard scaled Data
nb_classifier2 = GaussianNB()
nb_classifier2.fit(X_train_sc, y_train)
y_pred_nb_sc = nb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_nb_sc)


# # Decision tree

# In[47]:



from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
accuracy_score(y_test, y_pred_dt)


# In[48]:


# Train with Standard scaled Data
dt_classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier2.fit(X_train_sc, y_train)
y_pred_dt_sc = dt_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_dt_sc)


# # Random forest

# In[49]:


from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
accuracy_score(y_test, y_pred_rf)


# In[50]:


# Train with Standard scaled Data
rf_classifier2 = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier2.fit(X_train_sc, y_train)
y_pred_rf_sc = rf_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_rf_sc)


# # extratree classifier

# In[51]:



from sklearn.tree import ExtraTreeClassifier
xt_classifier = ExtraTreeClassifier(criterion = 'entropy', random_state = 51)
xt_classifier.fit(X_train, y_train)
y_pred_dt = xt_classifier.predict(X_test)
accuracy_score(y_test, y_pred_dt)


# In[52]:


#with standard scaled data
xt_classifier2 = ExtraTreeClassifier(criterion = 'entropy', random_state = 51)
xt_classifier2.fit(X_train_sc, y_train)
y_pred_dt_sc = xt_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_dt_sc)


# # Gradient boost

# In[53]:


from sklearn.ensemble import GradientBoostingClassifier
gb_classifier = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=4, max_depth=2, random_state = 51)
gb_classifier.fit(X_train, y_train)
y_pred_gb= gb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_gb)


# In[54]:


#with standard scaled data
from sklearn.ensemble import GradientBoostingClassifier
gb_classifier2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state = 51)
gb_classifier2.fit(X_train_sc, y_train)
y_pred_gb_sc= gb_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_gb_sc)


# # Adaboost

# In[55]:


from sklearn.ensemble import AdaBoostClassifier
adb_classifier = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                                    n_estimators=2000,
                                    learning_rate=0.1,
                                    algorithm='SAMME.R',
                                    random_state=1,)
adb_classifier.fit(X_train, y_train)
y_pred_adb = adb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_adb)


# In[56]:


# Train with Standard scaled Data
adb_classifier2 = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                                    n_estimators=2000,
                                    learning_rate=0.1,
                                    algorithm='SAMME.R',
                                    random_state=1,)
adb_classifier2.fit(X_train_sc, y_train)
y_pred_adb_sc = adb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_adb_sc)


# # XGBoost

# In[57]:


# XGBoost Classifier
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_xgb)


# In[58]:


# Train with Standard scaled Data
xgb_classifier2 = XGBClassifier()
xgb_classifier2.fit(X_train_sc, y_train)
y_pred_xgb_sc = xgb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_xgb_sc)


# XGBoost Parameter Tuning Randomized Search

# In[59]:


params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] 
}


# In[60]:


from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(xgb_classifier, param_distributions=params, scoring= 'roc_auc', n_jobs= -1, verbose= 3)
random_search.fit(X_train, y_train)


# In[61]:


random_search.best_params_


# In[62]:


random_search.best_estimator_


# In[63]:


# training XGBoost classifier with best parameters
xgb_classifier_pt = XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=0.3, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=0.3, gpu_id=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.3, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=3, max_leaves=None,
              min_child_weight=1, monotone_constraints=None,
              n_estimators=100, n_jobs=None, num_parallel_tree=None,
              predictor=None, random_state=None,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=3)

xgb_classifier_pt.fit(X_train, y_train)
y_pred_xgb_pt = xgb_classifier_pt.predict(X_test)


# In[64]:


accuracy_score(y_test, y_pred_xgb_pt)


# In[74]:


cm = confusion_matrix(y_test, y_pred_xgb)
plt.title('Heatmap of Confusion Matrix', fontsize = 15)
sns.heatmap(cm, annot = True)
plt.show()


# Classification report

# In[66]:


print(classification_report(y_test, y_pred_xgb))


# In[67]:


from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = xgb_classifier_pt, X = X_train_sc,y = y_train, cv = 10)
print("Cross validation accuracy of XGBoost model = ", cross_validation)
print("\nCross validation mean accuracy of XGBoost model = ", cross_validation.mean())


# grid search

# In[68]:


from sklearn.model_selection import GridSearchCV 
grid_search = GridSearchCV(xgb_classifier, param_grid=params, scoring= 'roc_auc', n_jobs= -1, verbose= 3)
grid_search.fit(X_train, y_train)


# In[69]:


grid_search.best_estimator_


# In[70]:


xgb_classifier_pt_gs = XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=0.5, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=0.2, gpu_id=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.2, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=3, max_leaves=None,
              min_child_weight=1, monotone_constraints=None,
              n_estimators=100, n_jobs=None, num_parallel_tree=None,
              predictor=None, random_state=None,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
xgb_classifier_pt_gs.fit(X_train, y_train)
y_pred_xgb_pt_gs = xgb_classifier_pt_gs.predict(X_test)
accuracy_score(y_test, y_pred_xgb_pt_gs)


# In[75]:


cm = confusion_matrix(y_test, y_pred_xgb_pt_gs)
plt.title('Heatmap of Confusion Matrix', fontsize = 15)
sns.heatmap(cm, annot = True)
plt.show()


# The model is giving 0 type II error so its the best

# In[73]:


import pickle

# save model
pickle.dump(xgb_classifier_pt_gs, open('breast_cancer_detector.pickle', 'wb'))

# load model
breast_cancer_detector_model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))

# predict the output
y_pred = breast_cancer_detector_model.predict(X_test)

# confusion matrix
print('Confusion matrix of XGBoost model: \n',confusion_matrix(y_test, y_pred),'\n')

# show the accuracy
print('Accuracy of XGBoost model = ',accuracy_score(y_test, y_pred))


# In[ ]:




