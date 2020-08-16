#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from scipy import io
from sklearn.model_selection import train_test_split
import time


# In[2]:


matfile = io.loadmat("mnist-original.mat")


# In[3]:


data = (pd.DataFrame(matfile['data'])).T
label = (pd.DataFrame(matfile['label'])).T
label.columns = ['y']


# In[4]:


df = pd.concat([data, label], axis=1)
df = df.sample(n=10000, random_state=1)


# # PCA

# In[5]:


from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[6]:


X = df.iloc[:, :-1]
y = pd.DataFrame(df.iloc[:, -1])


# In[7]:


ss = StandardScaler()
X_ss = ss.fit_transform(X)


# In[75]:


for i in range(len(X.columns)):
    model = PCA(n_components=i)
    pca_features = model.fit_transform(X_ss)
    print(str(i) + " : " + str(sum(model.explained_variance_ratio_)))


# In[ ]:


# 누적 설명률이 최초로 75%를 넘는 99을 n_components로 사용


# In[8]:


model = PCA(n_components=99)
X_pca = pd.DataFrame(model.fit_transform(X))


# # LDA

# In[9]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[90]:


for i in range(len(X.columns)):
    model = LinearDiscriminantAnalysis(n_components=i)
    lda_features = model.fit_transform(X_ss, y.values.ravel())
    print(str(i) + " : " + str(sum(model.explained_variance_ratio_)))


# In[ ]:


# 누적 설명률이 최초로 75%를 넘는 5를 n_components로 사용


# In[10]:


model = LinearDiscriminantAnalysis(n_components=5)
X_lda = pd.DataFrame(model.fit_transform(X, y))


# ### 비교

# In[11]:


X.head()


# In[12]:


X_pca.head()


# In[13]:


X_lda.head()


# In[14]:


X.describe()


# In[15]:


X_pca.describe()


# In[16]:


X_lda.describe()


# 속성의 갯수는 원본 데이터가 784개, pca가 99개, lda가 5개로 lda가 압도적으로 적다.
# 

# # 다항 분류기에 적용해보기

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca, y, test_size=0.2, random_state=1)
X_lda_train, X_lda_test, y_lda_train, y_lda_test = train_test_split(X_lda, y, test_size=0.2, random_state=1)


# ### Random Forest

# In[19]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# In[20]:


eclf = RandomForestClassifier(oob_score=True)
params ={
    "n_estimators" : [10, 100, 500],
    "max_features" : [10, 20, 50]
    }


# ##### 원본

# In[37]:


start = time.time()
grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5, n_jobs=-1)
grid = grid.fit(X_train, y_train.values.ravel())
print("WorkingTime : ", time.time() - start, "sec")


# In[38]:


# WorkingTime :  161.04402947425842 sec


# In[39]:


print("parameters : " + str(grid.best_params_))
print("score : " + str(grid.best_score_))
print("oob score : " + str(grid.best_estimator_.oob_score_))


# In[40]:


# parameters : {'max_features': 50, 'n_estimators': 500}
# score : 0.945875
# oob score : 0.947125


# In[41]:


start = time.time()
print("accuracy score : ", accuracy_score(y_test, grid.best_estimator_.predict(X_test)))
print("WorkingTime : ", time.time() - start, "sec")


# In[42]:


# accuracy score :  0.9485
# WorkingTime :  0.31615304946899414 sec


# #### pca

# In[43]:


params ={
    "n_estimators" : [10, 100, 500],
    "max_features" : [3, 5, 10, 20]
    }


# In[44]:


start = time.time()
grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5, n_jobs=-1)
grid = grid.fit(X_pca_train, y_pca_train.values.ravel())
print("WorkingTime : ", time.time() - start, "sec")


# In[45]:


# WorkingTime :  214.37747073173523 sec
# 더 길다.. 왜지?


# In[46]:


print("parameters : " + str(grid.best_params_))
print("score : " + str(grid.best_score_))
print("oob score : " + str(grid.best_estimator_.oob_score_))


# In[47]:


# parameters : {'max_features': 5, 'n_estimators': 500}
# score : 0.9255
# oob score : 0.9235


# In[48]:


start = time.time()
print("accuracy score : ", accuracy_score(y_pca_test, grid.best_estimator_.predict(X_pca_test)))
print("WorkingTime : ", time.time() - start, "sec")


# In[49]:


# accuracy score :  0.9235
# WorkingTime :  0.3441164493560791 sec


# #### lda

# In[50]:


params ={
    "n_estimators" : [10, 100, 500],
    "max_features" : [2,3,4,5]
    }


# In[51]:


start = time.time()
grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5, n_jobs=-1)
grid = grid.fit(X_lda_train, y_lda_train.values.ravel())
print("WorkingTime : ", time.time() - start, "sec")


# In[52]:


# WorkingTime :  75.84506487846375 sec


# In[53]:


print("parameters : " + str(grid.best_params_))
print("score : " + str(grid.best_score_))
print("oob score : " + str(grid.best_estimator_.oob_score_))


# In[54]:


# parameters : {'max_features': 2, 'n_estimators': 100}
# score : 0.854625
# oob score : 0.85075


# In[55]:


start = time.time()
print("accuracy score : ", accuracy_score(y_lda_test, grid.best_estimator_.predict(X_lda_test)))
print("WorkingTime : ", time.time() - start, "sec")


# In[56]:


# accuracy score :  0.8585
# WorkingTime :  0.06484055519104004 sec


# #### 결론
# 
# lda가 속도는 압도적으로 빠른 반면 정확도는 가장 떨어졌다. 
# 
# 다른 데이터의 경우 0.92~0.94 정도의 정확도를 보인 반면 0.859에 불과했다.
# 
# pca는 속도도 원본 데이터에 비해 느렸으며, 정확도 역시 크게 차이나진 않지만 원본 데이터에 비해 떨어졌다.
# 
# 원본 데이터는 가장 높은 정확도를 보였다.
# 
# 성능 : 원본 > PCA >>> LDA
# 
# 속도 : LDA >>> 원본 > PCA

# ### 나이브베이즈

# In[57]:


from sklearn.naive_bayes import GaussianNB


# In[58]:


gnb = GaussianNB()


# #### 원본

# In[70]:


start = time.time()
gnb.fit(X_train, y_train.values.ravel())
y_pred = gnb.predict(X_test)
print("accuracy score : ", accuracy_score(y_test, y_pred))
print("WorkingTime : ", time.time() - start, "sec")


# In[73]:


# accuracy score :  0.59
# WorkingTime :  0.2604246139526367 sec


# #### PCA

# In[71]:


start = time.time()
gnb.fit(X_pca_train, y_pca_train.values.ravel())
y_pred = gnb.predict(X_pca_test)
print("accuracy score : ", accuracy_score(y_pca_test, y_pred))
print("WorkingTime : ", time.time() - start, "sec")


# In[75]:


# accuracy score :  0.866
# WorkingTime :  0.04545259475708008 sec


# #### LDA

# In[72]:


start = time.time()
gnb.fit(X_lda_train, y_lda_train.values.ravel())
y_pred = gnb.predict(X_lda_test)
print("accuracy score : ", accuracy_score(y_lda_test, y_pred))
print("WorkingTime : ", time.time() - start, "sec")


# In[77]:


# accuracy score :  0.838
# WorkingTime :  0.005506277084350586 sec


# #### 결론

# 나이브 베이즈 분류기의 경우 PCA 혹은 LDA를 통해 차원축소를 한 경우가 압도적으로 높은 성능을 보였다.
# 
# 원본 데이터의 경우 0.59의 정확도를 보인 반면 PCA 데이터는 0.866, LDA 데이터는 0.838으로 매우 큰 차이를 보였다.
# 
# 속도의 경우 원본도 0.26초로 느리지 않았지만 PCA가 0.045초, LDA가 0.005초로 훨씬 더 빨랐다.
# 
# 성능 : PCA > LDA >>>>>>>>>>>>>>> 원본
# 
# 속도 : LDA > PCA > 원본
