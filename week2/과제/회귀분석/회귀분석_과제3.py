#!/usr/bin/env python
# coding: utf-8

# In[258]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor

plt.rc('font', family='NanumGothic')
plt.style.use('seaborn')
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[231]:


data = pd.read_csv('Auction_master_train.csv')


# ----
# EDA, 전처리

# In[232]:


data.isnull().sum()


# In[233]:


data.drop(['addr_dong','addr_li','addr_san','addr_bunji1','addr_bunji2','addr_etc','road_name','road_bunji1','road_bunji2','point.y', 'point.x', 'Specific'], axis=1, inplace=True)


# In[218]:


# 결측치가 대부분 '번지수' 와 관련된 데이터이며, 번지수가 타겟변수인 '최종낙찰가'와 아무런 관련이 없다고 충분히 생각할 수 있다.
# 주소와 관련된 데이터는 크게 밀접한 관련이 있을 것으로 예상할 수 있는 addr_do와 addr_si 까지만 살리고, 
# 그 이하를 나타내는 속성은 모두 제거했다.
# 또한 특이사항 역시 회귀분석에는 적합하지 않은 변수이므로 제거했다.


# In[219]:


data.isnull().sum()


# ----
# 범주형 변수 인코딩하기

# In[234]:


print(data['Apartment_usage'].unique())
print(data['Bid_class'].unique())
print(data['addr_do'].unique())
print(data['addr_si'].unique())
print(data['Auction_class'].unique())
print(data['Share_auction_YorN'].unique())
print(data['Close_result'].unique())

#7가지 변수 모두 범주형 변수지만 명목형이다. 
#주소 관련 변수는 타겟변수인 hammer_price와 상관관계가 있을 것으로 보이는 두가지 변수를 선택했다.
#Apartment_usage, Bid_class, addr_do, addr_si 는 모두 one-hot encoding으로 처리하고, 
#Auction_class, Share_auction_YorN, Close_result 는 값이 2개만 존재하므로 1과 0으로 처리한다.
#
#이때 직관성을 높이기 위해 Auction_class는 forced_auction 으로, Close_result는 is_closed로, Share_auction_YorN은 share_auction으로 속성 이름을 변경한다.


# In[235]:


dummy_var = pd.get_dummies(data.Apartment_usage)
data = pd.concat([data.drop(['Apartment_usage'], axis=1),dummy_var], axis=1)

dummy_var = pd.get_dummies(data.Bid_class)
data = pd.concat([data.drop(['Bid_class'], axis=1),dummy_var], axis=1)

dummy_var = pd.get_dummies(data.addr_do)
data = pd.concat([data.drop(['addr_do'], axis=1),dummy_var], axis=1)

dummy_var = pd.get_dummies(data.addr_si)
data = pd.concat([data.drop(['addr_si'], axis=1),dummy_var], axis=1)


# In[236]:


data['Auction_class'] = data['Auction_class'].replace('임의', 0)
data['Auction_class'] = data['Auction_class'].replace('강제', 1)

data['Share_auction_YorN'] = data['Share_auction_YorN'].replace('N', 0)
data['Share_auction_YorN'] = data['Share_auction_YorN'].replace('Y', 1)

data['Close_result'] = data['Close_result'].replace('배당', 1)
data['Close_result'] = data['Close_result'].replace('    ', 0)


# In[238]:


data.rename(columns={data.columns[1] : 'force_auction',
                    data.columns[21] : 'share_auction',
                    data.columns[23] : 'is_closed'}, inplace=True)


# In[239]:


print(data.columns.values)


# ----
# EDA

# In[267]:


lm = sns.lmplot(x="Claim_price", y="Hammer_price", data=data)
axes = lm.axes
axes[0,0].set_ylim(0,1000000000)
axes[0,0].set_xlim(0,1000000000)


# In[268]:


#Claim_price와 Hammer_price는 아주 약한 상관관계를 보이고 있다.
#대체로 경매 신청인의 청구금액보다 낙찰가가 더 높은 모습을 보인다. 


# In[269]:


lm = sns.lmplot(x="Total_building_auction_area", y="Hammer_price", data=data)
axes = lm.axes
axes[0,0].set_ylim(0,5500000000)
axes[0,0].set_xlim(0,300)


# In[270]:


#건물의 총 경매 면적인 Total_building_auction_area와는 비교적 높은 상관관계를 보이고 있다.
#면적이 넓을수록 매매가가 대체로 높다는 의미로 당연한 결과라고 볼 수 있다.


# In[271]:


lm = sns.lmplot(x="Minimum_sales_price", y="Hammer_price", data=data)
axes = lm.axes
axes[0,0].set_ylim(0,3500000000)
axes[0,0].set_xlim(0,3500000000)


# In[272]:


#Minumum sales price와 타겟변수는 서로 거의 일치하는 결과를 나타내고 있다.


# In[273]:


lm = sns.lmplot(x="Total_floor", y="Hammer_price", data=data)
axes = lm.axes
axes[0,0].set_ylim(0,3500000000)
axes[0,0].set_xlim(0,80)


# In[274]:


#해당 건물의 층수를 의미하는 Total_floor와는 약한 상관관계를 보이고 있다. 
#즉 고층건물일수록 매매가가 비싼 경향은 있지만 상관관계는 0.2로 약하다.


# In[275]:


lm = sns.boxplot(x="Auction_count", y="Hammer_price", data=data)
lm.set_ylim(0,3000000000)
lm.set_xlim(-1,7)


# In[276]:


#총경매횟수와는 아무런 관계가 없다고 보는 것이 타당해보인다.


# In[284]:


sns.violinplot("서울","Auction_count", hue="아파트", data=data,split=True).set_ylim(0, 6)
plt.show()


# In[285]:


#부산과 서울의 총경매횟수를 아파트 / 주상복합으로 구분했다.
#파란색이 주상복합, 초록색이 아파트를 의미하고 왼쪽이 부산, 오른쪽이 서울이다.
#총경매횟수는 큰 의미가 없어보이며, 서울이 주상복합의 비중이 더 높다.


# In[291]:


sns.factorplot('force_auction','Hammer_price',hue='서울',data=data)
plt.show()


# In[292]:


#부산과 서울을 강제경매/임의경매로 구분해서 최종 경매가를 나타냈다.
#초록색 선이 서울, 파란색 선이 부산을 나타낸다. x값이 0인 경우 임의 경매를 의미하며 1인 경우 강제 경매를 의미한다.
#서울이 부산보다 최종 경매가가 높으며, 임의경매일 경우가 강제경매일 경우보다 매매가가 높다.


# In[360]:


lm = sns.relplot(x="Total_floor", y="Hammer_price", hue="서울", data=data, s=15, x_jitter=100)
axes = lm.axes
axes[0,0].set_ylim(0,3500000000)
axes[0,0].set_xlim(0,80)


# In[353]:


#초록색 점이 서울, 파란색 점이 부산을 의미하며, y축은 최종 경매가, x축은 건물의 층수를 의미한다.
#서울이 경매 건수가 훨씬 많으며 경매가가 높을수록 서울의 비중이 높다.
#단 고층 건물로 갈수록 부산의 비중이 더 높아지고 있다.


# ----
# 회귀분석

# 변수선택

# In[240]:


## Auction_key : Key값이므로 제외
## forced_auction : 최종 낙찰가에 영향을 줄 것으로 예상되므로 포함
## Claim_price : 최종 낙찰가에 영향을 줄 것으로 예상되므로 포함
## Appraisal_company, Creditor : 많은 값을 가지고 있는 명목형 변수이고, 낙찰가에 큰 영향을 주지 않을 것으로 예상되므로 제외
## Auction_count : EDA 결과 거의 영향을 미치지 않을 것으로 보이므로 제외
## Auction_miscarriage_count : 마찬가지 이유로 제외
## 면적 관련 변수 : 영향이 있을 것으로 보이므로 포함
## Total_appraisal_price : EDA 결과 Hammer price와 매우 높은 상관관계가 있으므로 포함하면 정확도가 높아지겠지만 실습을 위해 제외
## Minimum_sales_price : EDA 결과 Hammer price와 매우 높은 상관관계가 있으므로 포함하면 정확도가 높아지겠지만 실습을 위해 제외
## 날짜 관련 변수 : 전부 제외함
## Total_floor, Current_floor : 포함
## is_closed, share_auction : 포함
## Final_result : 전부 같은 값을 지니고 있으므로 제외
## 앞에서 인코딩한 데이터들 전부 포함, 단 '구' 를 나타내는 데이터는 지나치게 컬럼이 많아져 분석에 어려움이 생겨 제거


# In[249]:


reg_data = data[['force_auction','Claim_price','Total_land_gross_area', 'Total_land_real_area', 'Total_land_auction_area', 'Total_building_area', 'Total_building_auction_area', 'Total_floor', 'Current_floor', 'is_closed', 'share_auction', '아파트', '주상복합', '개별', '일괄', '일반', '부산', '서울', 'Hammer_price']]


# In[250]:


reg_x = reg_data.drop(['Hammer_price'], axis=1)
reg_y = pd.DataFrame(reg_data['Hammer_price'], columns=['Hammer_price'])


# In[251]:


print(reg_x.columns.values)
print(reg_y.columns.values)


# In[257]:


plt.figure(figsize=(10,10))
sns.heatmap(data = reg_x.corr(), annot=True, 
fmt = '.2f', linewidths=.5, cmap='Blues')


# In[261]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(
    reg_x.values, i) for i in range(reg_x.shape[1])]
vif["features"] = reg_x.columns
vif.sort_values(["VIF Factor"], ascending=[False])


# In[ ]:


#Total_land_gross_area를 제외한 면적을 나타내는 4개의 변수가 다중공선성이 매우 높다고 볼 수 있으므로, Total_building_auction_area만 선택했다.
#범주형 변수를 인코딩한 변수들은 무한값이 나왔는데 무시하고 회귀분석에 이용할 수 있다.


# In[264]:


reg_x.drop(["Total_land_auction_area", "Total_land_real_area", "Total_building_area"], axis=1, inplace=True)


# ----
# 모델링

# In[282]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(reg_x, reg_y, test_size=0.2, random_state=0)


# In[283]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_train, y_train)


# In[270]:


#MSE
import sklearn as sk
sk.metrics.mean_squared_error(y_train, model.predict(X_train))


# In[271]:


print(model.coef_)
print(model.intercept_)


# In[272]:


model.predict(X_test)


# In[273]:


model.score(X_test, y_test)


# In[274]:


y_pred = model.predict(X_test) 
plt.plot(y_test, y_pred, '.')

# 예측과 실제가 비슷하면, 라인상에 분포함
x = np.linspace(0, 330, 100)
y = x
plt.plot(x, y)
plt.show()


# In[276]:


## MSE : 9.480804715057722e+16
## Train R-Square : 0.73
## Test R-Square : 0.53

## MSE가 매우 높다.. 매우..


# ----
# Ridge, Lasso

# In[288]:


from sklearn.linear_model import Ridge, Lasso

ridge=Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge.score(X_test,y_test)


# In[289]:


ridge=Ridge(alpha=0.3)
ridge.fit(X_train, y_train)
ridge.score(X_test,y_test)


# In[290]:


sk.metrics.mean_squared_error(y_train, ridge.predict(X_train))


# In[291]:


lasso=Lasso(alpha=0.3)
lasso.fit(X_train, y_train)
lasso.score(X_test, y_test)


# In[ ]:


## 정규화 정도나 모델에 상관없이 R-square 값과 MSE값 모두 거의 같은 값을 보이므로 큰 의미는 없어보인다.

