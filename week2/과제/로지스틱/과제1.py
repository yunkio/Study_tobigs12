#!/usr/bin/env python
# coding: utf-8

# # Sigmoid_Function_Overview
# - 본 Tutorial은 가천대학교 산업경영공학과 최성철 교수님의 Machine Learning 강의를 바탕으로 작성되었습니다.

# In[8]:


import matplotlib.pyplot as plt
import numpy as np
import math


# In[16]:


# Sigmoid function, Logistic Function 구현

def sigmoid(x):
    a = []
    for item in x:
        a.append(1 / (1+math.exp(-item)))  # 우리가 알고있는 Sigmoid Function을 
                                           # 코드로 표현하면 이렇게 표현됩니다.
    return a


# In[10]:


x = np.arange(-10., 10., 0.2)
sig = sigmoid(x)
plt.plot(x, sig)
plt.show()


# In[11]:


#Odds 구현

p = np.arange(0., 1., 0.01)
odds = p/(1-p)
plt.plot(p, odds)
plt.show()


# In[12]:


# P값이 1에 가까워 질수록 Odds 값이 무한대로 증가하는 것을 알 수 있다.
# 반대로 P값이 0에 가까워질수록 Odds값은 0에 가까워진다.


# In[13]:


def log_odds(p):
    return np.log(p/(1 - p))

x = np.arange(0.005, 1, 0.005)
log_odds_x = log_odds(x)
plt.plot(x, log_odds_x)
plt.ylim(-8, 8)
plt.xlabel('x')
plt.ylabel('log_odds(x)')

# y axis ticks and gridline
plt.yticks([-7, 0, 7])
ax = plt.gca()
ax.yaxis.grid(True)

plt.tight_layout()
plt.show()


# ### Q1. 위의 그래프를 어떻게 해석하면 좋을까?
# - P값이 1에 가까워 질수록 ...
# - P값이 0에 가까워 질수록 ...

# P값이 1에 가까워 질수록 범주 1에 속할 확률이 점점 높아진다.
# 
# 반대로 P값이 0에 가까워 질수록 범주 1에 속할 확률이 점점 낮아진다, 즉 범주 0에 속할 확률이 높아진다.

# In[14]:


x = np.arange(0.005, 1, 0.005)
y = -np.log(1-x)
plt.plot(x,y)
plt.show()


# ### Q2. Logit 값에 역산을 취해주면 어떻게 될까?

# 입력값을 통해 얻은 Logit 값을 통해 회귀식을 추정한 뒤, 역산을 취해 성공확률을 구할 수 있는 계수를 얻는다.
# 
# 이 계수를 활용해 분류 값을 예측한다.

# In[ ]:




