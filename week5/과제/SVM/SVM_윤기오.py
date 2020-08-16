#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV


# In[3]:


df = pd.read_csv('./train.csv')


# In[4]:


df.shape


# In[77]:


data = df.sample(n=10000, random_state=1)
X = data.iloc[:, 2:]
y = data.iloc[:, 1:2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[31]:


param_grid = {'C' : [0.001, 0.01, 0.1, 1, 10], 'gamma' : [0.001, 0.01, 0.1, 1]}
grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train, y_train.values.ravel())


# In[32]:


print(grid_search.best_score_)
print(grid_search.best_params_)


# In[ ]:


# 0.907375
# {'C': 0.001, 'gamma': 0.001}


# In[78]:


from sklearn.svm import SVC
from sklearn import metrics

svc = SVC(C=0.001, gamma=0.001, kernel='rbf')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))


# In[ ]:


# Accuracy Score:
# 0.9005


# ### PCA 후 적용

# In[53]:


from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# In[ ]:


# 이미 표준화가 된 데이터임


# In[60]:


for i in range(len(X.columns)):
    model = PCA(n_components=i)
    pca_features = model.fit_transform(X)
    print(str(i) + " : " + str(sum(model.explained_variance_ratio_)))


# In[ ]:


# 최초로 누적설명률이 75%가 넘는 54로 n_components를 선택


# In[65]:


model = PCA(n_components=54)
pca_features = model.fit_transform(X)
X = pd.DataFrame(pca_features)


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[ ]:


param_grid = {'C' : [0.001, 0.01, 0.1, 1, 10], 'gamma' : [0.001, 0.01, 0.1, 1]}
grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train, y_train.values.ravel())


# In[ ]:


print(grid_search.best_params_)
print(grid_search.best_score_)


# In[ ]:


# 0.907375
# {'C': 0.001, 'gamma': 0.001}
# 위와 같이 C값과 gamma값이 grid의 가장 낮은 값으로 선정되었음
# kernel을 linear로 한 값도 평가할 필요가 있어보임


# In[82]:


param_grid = {'C' : [0.00001, 0.0001, 0.001]}
grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=5)
grid_search.fit(X_train, y_train.values.ravel())


# In[83]:


print(grid_search.best_params_)
print(grid_search.best_score_)


# In[ ]:


# {'C': 1e-05}
# 0.907375
# 다 같은 점수(정확도)로 평가된다.. ㅠㅠ


# In[91]:


from sklearn.svm import SVC
from sklearn import metrics

svc = SVC(C=1e-05, kernel='linear')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))


# In[92]:


# Accuracy Score:
# 0.9005

# 정확도 역시 같다..


# In[ ]:




