#!/usr/bin/env python
# coding: utf-8

# # German Credit Dataset
# - 대출인지 아닌지를 예측하는 문제
# - 데이터를 NB에 맞도록 간단하게 변환합니다.
# - Binary 데이터들로 이루어진 대출 사기 데이터들로 부터 대출인지 아닌지 예측해보세요.

# In[101]:


from pandas import Series, DataFrame
import pandas as pd
import numpy as np


# In[102]:


data_url = './fraud.csv'
df = pd.read_csv(data_url, sep=',')
df.head()


# In[103]:


# ID열을 삭제해줍니다.
del df["ID"] 

# Label(Y_data)을 따로 저장해 줍니다.
Y_data = df.pop("Fraud")


# In[104]:


# as_matrix()함수를 통해 array형태로 변환시켜 줍니다.
# Convert the frame to its Numpy-array representation.
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.as_matrix.html

Y_data = Y_data.as_matrix()
Y_data


# In[105]:


type(Y_data)


# In[106]:


df.head()


# In[107]:


# 우리가 앞으로 사용할 데이터 셋입니다. 그런데 문제가 있어보이네요...


# ## One-Hot encoding

# * 범주형 변수를 dummy변수로 변환해주는 작업
# * Do it yourself!

# ### 1. Do One-Hot Encoding! 

# In[108]:


df


# In[109]:


# 범주형 변수 처리 문제입니다.
# 앞선 EDA 시간과 Logistic EDA를 통해 우리는 범주형 변수를 처리해 주는 방법을 배웠습니다.
# get_dummies를 사용해서 One-Hot encoding 처리를 해주세요.

x_df = pd.get_dummies(df, prefix=['History', 'CoApplicant', 'Accommodation'])
x_df.head() # dummy변수로 변환


# * One-Hot Encoding이 제대로 되었다면 우리는 10개의 Feature를 얻을 수 있습니다.

# In[110]:


x_data = x_df.as_matrix()
x_data


# In[111]:


# one-hot encoding을 통해 10개의 Feature를 얻었다.


# #### Q1. as_matrix()함수를 통해 우리가 하고자 하는 것은 무엇일까요? 

# 나이브 베이즈를 적용하기 위해서는 데이터를 행렬 형태로 구성할 필요가 있다.

# In[112]:


Y_data == True # boolean index


# In[113]:


len(set(Y_data))


# ## Naive bayes classifier

# * P(Y)
# * P(X1, X2, ..., Xn)
# * P(Y|X1, X2, X3, ..., Xn)
# * P(X1|Y), P(X2|Y), ... P(Xn|Y)
# 등 우리가 구해야 할 식들에 대한 아이디어가 있어야 합니다.

# ### P(Y1), P(Y0) 구하기

# In[114]:


# P(Y1), P(Y0)
# P(Y1) = count(Y1) / count(Y)

P_Y_True = sum(Y_data==True) / len(Y_data)
P_Y_False = 1 - P_Y_True

P_Y_True, P_Y_False


# * 이번 튜토리얼에서는 **index를 이용합니다.**
# * 이해하기보다는 따라 하면서 음미해보세요.

# In[115]:


# y가 1일 경우, y가 0일 경우를 구해줘야 합니다.
# 이번 시간에는 np.where를 사용합니다.
# np.where


# In[116]:


ix_Y_True = np.where(Y_data) # Y_data == True인 인덱스를 뽑아줍니다.
ix_Y_False = np.where(Y_data==False)

ix_Y_True, ix_Y_False


# In[117]:


# np.where을 사용해서 Y가1일 때와 0일 때 각각의 인덱스 값을 얻을 수 있게 되었습니다.


# In[118]:


# P(X|Y) = count(X_cap_Y) / count(Y)


# ### productP(X|Yc) 구하기
# 
# * product * P(X|Y1)
# * product * P(X|Y2)

# In[153]:


p_x_y_true = x_data[ix_Y_True].sum(axis=0) / sum(Y_data == True)
p_x_y_true


# In[155]:


p_x_y_true = x_data[ix_Y_True].sum(axis=0) / sum(Y_data==True)  # Q.뒤에 sum(Y_data == True) 필요한가요? # 앞에 식이 P(X_cap_Y1)인 것 같은데...
p_x_y_false = x_data[ix_Y_False].sum(axis=0) / sum(Y_data==False)

p_x_y_true, p_x_y_false


# In[139]:


# 총 10개의 값에 대해서 확률을 구해준다.


# In[209]:


x_test = [0,1,0,0,0,1,0, 0,1,0]

import math

x_cap_y_true = np.array([p_x_y_true[i] if x_test[i] == 1 else 1-p_x_y_true[i] for i in range(len(x_test))])
x_cap_y_false = np.array([p_x_y_false[i] if x_test[i] == 1 else 1-p_x_y_false[i] for i in range(len(x_test))])
num = 0.00001

p_y_true_test = math.log(P_Y_True) + sum(np.log(x_cap_y_true+num))
p_y_false_test = math.log(P_Y_False) + sum(np.log(x_cap_y_false+num))

p_y_true_test, p_y_false_test


# In[207]:


p_y_true_test < p_y_false_test


# ## 2. Do Smoothing을 통해 P(Y=1|X)의 확률과 P(Y=0|X)의 확률 값을 비교하세요.

# In[134]:


smoothing_p = 2


# In[218]:


import math
x_test = [0,1,0,0,0,1,0, 0,1,0]

p_x_y_true = ((x_data[ix_Y_True].sum(axis=0) + smoothing_p) / (sum(Y_data==True) + smoothing_p * len(set(Y_data))))
p_x_y_false = ((x_data[ix_Y_False].sum(axis=0) + smoothing_p) / (sum(Y_data==False) + smoothing_p * len(set(Y_data))))

x_cap_y_true = np.array([p_x_y_true[i] if x_test[i] == 1 else 1-p_x_y_true[i] for i in range(len(x_test))])
x_cap_y_false = np.array([p_x_y_false[i] if x_test[i] == 1 else 1-p_x_y_false[i] for i in range(len(x_test))])

p_y_true_test = math.log(P_Y_True) + sum(np.log(x_cap_y_true))
p_y_false_test = math.log(P_Y_False) + sum(np.log(x_cap_y_false))

p_y_true_test, p_y_false_test


# In[222]:


x_data[ix_Y_True].sum(axis=0)


# In[215]:


p_y_true_test < p_y_false_test


# # 결과값에 대한 설명과 해석을 달아주세요

# p_y_true_test 가 p_y_false_test 보다 높다는 것을 알 수 있다.
# 
# 따라서 x_test는 목표변수 Y인 'Fraud' 가 True가 나올 확률이 더 높다.

# In[ ]:




