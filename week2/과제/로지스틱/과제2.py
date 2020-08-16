#!/usr/bin/env python
# coding: utf-8

# # Money Ball - Basic EDA & Modeling(Logistic)
# 
# ### 2주차 과제는 Money Ball Data Analysis입니다. 
# ### 우리가 Binary Classification하고자 하는 변수는 play-off입니다.
# ### Money Ball Data Set을 분석하고 어느 팀이 play-off에 진출하는지 
# ### Logistic Regression방법을 통해 분석합니다.
# _How does a team make the playoffs?_
# 
# _How does a team win more games?_
# 
# _How does a team score more runs?_

# - 머니볼과 빅데이터
# - http://writting.co.kr/2015/04/%EB%A8%B8%EB%8B%88%EB%B3%BC-%EA%B7%B8%EB%A6%AC%EA%B3%A0-%EB%B9%85%EB%8D%B0%EC%9D%B4%ED%84%B0/

# - _이번 과제를 통해 우리는 어떤 팀이  play-off(가을야구)에 진출하는지 로지스틱 모델을 통해 분석합니다._
# + _Ipython 파일의 빈 부분을 채워주세요._
# + _하나하나 천천히 따라와 주세요._ 
# 
# + _W(Wins) Feature를 제외한 Feature들 중에서 가을야구 진출에 가장 영향을 많이 주는 Feature는 무엇일까요?_
# + _통념이 만연하던 야구의 편견을 깬 Money Ball은 과연 무엇일까요?_

# ### What is Money Ball?
# - Billy Bean & DePodesta's Story.
# 
# In the early 2000s, Billy Beane and Paul DePodesta worked for the Oakland Athletics. While there, they literally changed the game of baseball. They didn't do it using a bat or glove, and they certainly didn't do it by throwing money at the issue; in fact, money was the issue. They didn't have enough of it, but they were still expected to keep up with teams that had much deeper pockets. This is where Statistics came riding down the hillside on a white horse to save the day. This data set contains some of the information that was available to Beane and DePodesta in the early 2000s, and it can be used to better understand their methods.

# # 1. Import Library
# 
# - 앞으로 데이터 분석 및 모델링을 함에 있어서 첫 스텝은 필요한 Library를 불러오는 것입니다.

# In[390]:


# python library import

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("./"))


# In[391]:


from sklearn.linear_model import LogisticRegression # sklearn을 사용하여 Logistic 회귀분석을 할 경우 필요
import matplotlib.pyplot as plt # 시각화를 위한 library
import warnings
warnings.filterwarnings('ignore')

## Jupyter Notebook 이나 ipython 을 사용하다보면 향후 버전이 올라갈 때 변경될 사항 등을 알려주는 경고 메시지(warning message)가 거슬릴 때가 있습니다.
## 출처: https://rfriend.tistory.com/346 [R, Python 분석과 프로그래밍 (by R Friend)]


# # 2. Load Data & Data Exploration 

# * dataset 불러오기 
# * pandas를 이용해서 CSV파일을 불러오세요
# * 불러온 데이터를 파악해봅니다

# In[392]:


data = pd.read_csv("./baseball.csv")
data.head() # pandas로 data를 불러오면 습관적으로 head()를 찍어봅니다!


# In[393]:


# 주어진 데이터 셋이 어떤 데이터인지 한번 쭉 살펴봅니다.


# * Tip. 우리가 일반적으로 알고 있는 데이터 셋이 아닐 경우 각각의 Feature(Column, Attribute)가 무엇을 의미하는지 이해할 필요가 있습니다.
# **우리는 앞으로 Feature로 통일하겠습니다**
# - Data Set에 대한 설명을 참조합니다.
# - https://www.kaggle.com/wduckett/moneyball-mlb-stats-19622012

# ## 2-1. 각각의 데이터가 무엇을 의미하는지 파악하기
#     - 이미 알고 있는 내용이라면 건너 뛰셔도 됩니다.
#     e.g) Team: Major League Team 이름이구나.
#          League: 소속 League를 말하는구나.
#          Year: 데이터가 기록된 년도를 의미하는구나.
#          Rs: (Runs Scored) 득점 스코어를 의미하는구나
#          RA: (Runs Allowed) 실점스코어를 의미하는구나
#          .
#          .
#          .

# In[394]:


# 어떤 Feature가 있을까?
data.columns


# In[395]:


# Feature는 몇 개일까?
len(data.columns)


# * 특정 feature는 종속 변수에 아무런 영향을 주지 않을 수 있습니다. 
# * 그런 feature들을 파악하고 제거한다면 우리의 모델은 더욱 정확해 집니다.
# * 하지만 마냥 변수를 제거할 수도 없는 노릇입니다. 
# * 지난 1주차에 배웠던 EDA와 Preprocessing을 참고하여 데이터를 분석해봅시다.

# ## Data Exploration

# In[396]:


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# 각종 시각화 패키지 불러오기


# In[397]:


# for using Korean font
matplotlib.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False


# In[398]:


display(data.info())


# In[399]:


# 1232개의 entries, 15개의 column
# null 값이 있는지 isnull.sum()으로 확인을 해보자.


# In[400]:


data.isnull().sum()


# In[401]:


# 어떻게 처리할지 생각해보자


# # Q. Null 값을 어떻게 전처리 할 것인가?
# - 데이터를 본격적으로 분석 하기 전에 한번 생각해보도록 합니다.

# In[402]:


display(data.describe())


# In[403]:


# 이산변수 W


# ## 2-2.  변수 종류 확인하기

# In[404]:


data.info()


# ###  1) 범주형 변수 확인하기

# In[405]:


# categorical variable
categorical_col = list(data.select_dtypes(include='object').columns)
categorical_col


# In[406]:


# play-off Feature의 경우 0과 1로 범주형이지만 데이터에는 int type로 저장되어있다.


# ###  2) 연속형 변수 확인하기

# In[407]:


# numerical variable
numerical_col =  list(data.select_dtypes(include=('int64', 'float64')).columns)
numerical_col


# In[408]:


len(numerical_col)


# ### 3) 변수 종류 확인
# - 2개의 categorical variable(Team, League)와 13개의 numerical variable

# ###  4) 각각의 변수에 어떤 값이 들어있을까?
# - 각 변수별 unique값을 찍어본다.

# In[409]:


# for categorical_col

for col in categorical_col:
    print(col + ': ', len(set(data[str(col)])))


# In[410]:


# 39개의 팀과 2개의 리그
# 리그는 지난 수십년간 2개였다.


# In[411]:


# for numerical_col

for col in numerical_col:
    print(col + ': ', len(set(data[str(col)])))


# In[412]:


# 지난 수십년은 46년이었다.
# Playoffs가 2인 것으로 보아 categorical 변수로 바꿔줘도 무관할 것 같다.
# Q. G(Games Played)는 어떤 값을 가지고 있을까?


# In[413]:


data.G.head()


# In[414]:


# Game 수이다. 한 시즌에 치뤄진 경기수를 나타낸다.


# In[415]:


data.G.mean()


# In[416]:


# 47년간 평균 161.918경기가 치뤄졌다. 


# # 2. Data Preprocessing 

# ### T1. Column 삭제하기
# - W(Wins), 승리 외에 팀의 가을야구 진출에 영향을 많이 미치는 Feature가 알고 싶습니다.
# - del or drop을 사용하여 W column을 삭제해주세요

# In[417]:


data.head()


# In[418]:


del data['W']
data.head()


# In[419]:


# W Column을 지워줬다.


# ### Task2. 인코딩: League
# - League Feature는 AL과 NL로 이루어져 있습니다.
# - 지난 시간에 배웠던 인코딩 방법을 적용해서 모델이 학습 할 수 있도록 처리해주세요.
# - replace()함수를 사용합니다.

# In[420]:


set(data.League)


# In[421]:


data.League.replace({'AL':0, 'NL':1}, inplace=True)


# In[422]:


data.League.head()


# ### Task3. column 삭제하기
# - Team column을 삭제해주세요.
# - Team column이 없어도 모델에는 큰 영향이 없을 것 같습니다.

# In[423]:


data.head()


# In[424]:


del data['Team']
data.head()


# ### Task4. NaN값 처리하기
# - head를 찍어보니 NaN값이 보입니다. 
# - NaN값을 처리해 줍니다.

# In[425]:


data.isnull().sum()


# In[426]:


# RankSeason, RankPlayoffs, OOBP, OSLG 변수에 Null 값이 있습니다.
# 우리는 위의 변수들에 NaN값을 처리해줄 것입니다.


# data.RankSeason.head()

# In[427]:


data.RankSeason.head()


# In[428]:


data.RankPlayoffs.head()


# In[429]:


# OOBP는 Opponent On-Base Percentage. 
# OSLG는 Opponent Slugging Percentage.

data.describe()


# * OOBP와 OSLG값에 평균 값을 넣어주도록 합니다.
# * sklearndml SimpleImputer를 사용합니다.
# * 자세한 내용은 https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html

# In[430]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=0)
imputer = imputer.fit(data[['OOBP', 'OSLG']])

data[['OOBP', 'OSLG']] = imputer.transform(data[['OOBP', 'OSLG']])


# In[431]:


data.OOBP.isnull().sum()


# In[432]:


data.OSLG.isnull().sum()


# In[433]:


data.RankPlayoffs.isnull().sum()


# In[434]:


data.RankSeason.isnull().sum()


# In[435]:


del data['RankPlayoffs']
del data['RankSeason']


# In[436]:


data.head()


# In[437]:


del data['Year']


# In[438]:


data.head()


# # 3. train_test_split

# 데이터 셋을 독립변수와 종속변수로 나눠준다.varialbe(X) and dependent(y) variable.

# In[439]:


X = data.drop(["Playoffs"], axis=1)
y = pd.DataFrame(data["Playoffs"])


# splitting the data set as test_set and train_set to make predictions usoing the calssfiers.

# In[440]:


X.head()


# In[441]:


# Feature G를 삭제해줍니다

del X['G'] 


# In[442]:


X.head()


# * Splitting the data set as test_set and train_set to make predictions using the classifiers.

# In[443]:


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # Q1. train_tset_split module
# * train_test_split() 함수에 들어가는 각각의 인자 값은 무엇을 의미하는가?
# 
# - arrays
# - test_size
# - random_state
# 

# In[ ]:


#arrays : train과 test 데이터로 나눌 데이터들을 의미한다.
#test_size : 전체 데이터 중 어느정도 비율을 테스터 데이터로 나눌 것인지를 의미한다.
#random_state : 시드값을 의미한다.


# # 4. Feature Scaling
# 
# 경우에 따라 값이 특정 범위에서 매우 높은 범위로 변환되어 피쳐 스케일링을 사용합니다.

# In[444]:


# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
warnings.filterwarnings(action='once')


# ## Q2. Scaling
# Scaling을 통해 우리가 하고자 하는 것은 무엇인가요? 

# In[ ]:


#평균이 0, 표준편차가 1이 되도록 데이터값을 다시 스케일링한다. 
#스케일링을 하지 않으면 모델에 이상이 발생할 수 있다.


# # 5. Modeling 

# ## Q3. LogisticRegression() 모델을 만들어주세요. 그리고 만든 모델 인자값에 들어가는 값들의 의미를 설명해주세요.
# - e.g LogisticRegression(random_state=0, solver='', multi_class='')
# 
# - **모델 명은 model로 만듭니다**
# - random_state
# - solver
# - multi_class

# In[453]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=None, solver="liblinear", multi_class="auto")
#random_state는 시드값을 의미한다. 
#solver는 최적화에 사용되는 알고리즘을 의미한다.
#multi_class는 목표 변수의 class 갯수가 3개 이상일 때 사용되는 인자로 ovr의 경우 각 class 별로 해당 class vs 나머지 class로 
#binary decision을 내린다. multinomial의 경우 전체 확률 분포에 걸쳐 multinomial loss를 적용한다.

#데이터가 그렇게 크지 않으므로 solver는 liblinear를 선택하고, 이때 multi_class는 자동으로 ovr를 사용하게 된다.


# ## Q4. data를 교차검증 해주세요.(10-fold cross_validation)
# 
# - 10-fold cross_validation을 위한 인자값을 입력해주세요.
# - kfold = selection.KFold("교차검증을 위한 인자 만들기")
# - 교차검증 결과를 출력하고 해석합니다.
# 

# In[463]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=False, random_state=None)


# In[464]:


X_test = pd.DataFrame(X_test)
X_train = pd.DataFrame(X_train)
y_test = pd.DataFrame(y_test)
y_train = pd.DataFrame(y_train)


# In[469]:


predication = []
for train, test in kf.split(X_train):
    train_data=X_train.iloc[train]
    train_prad=y_train.iloc[train]
    model.fit(train_data,train_prad.values.ravel())
    test_prad=model.predict(X_train.iloc[test])
    predication.append(test_prad)
    
    
predication = np.concatenate(predication, axis=0)
predication[predication>0.5] = 1
predication[predication<=0.5] = 0

accuracy = sum(predication==y_train['Playoffs'])/len(predication)
accuracy
    


# ## 6.  Feature Selection
# - ref: https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python

# - 중요 Feature를 선택하는 방법입니다.
# - Kaggle 자료를 참고하였습니다.
# - 내가 만든 모델에서 어떤 변수가 중요한지 한 번 살펴보세요

# In[363]:


X.head()


# In[364]:


data.head()


# In[366]:


from sklearn.feature_selection import RFE

cols = ["BA", "League", "OOBP", "OSLG", "RA", "RS", "SLG"]
X = data[cols]
y = data['Playoffs']

# Build a logreg and compute the feature importances
model = LogisticRegression(random_state=None, solver="liblinear", multi_class="auto").fit(X_train, y_train.values.ravel())
# create the RFE and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(X, y)
# summarize the selection of the attributes
print('Selected features: %s' % list(X.columns[rfe.support_]))


# 중요 Feature를 3개 정도 뽑아봤는데
# Batting Average가 포함되지 않았다. 신기하다.

# Q. How to calculate Odds ratio?

# https://stackoverflow.com/questions/38646040/attributeerror-linearregression-object-has-no-attribute-coef

# In[367]:


model.fit(X, y)
model.coef_


# In[368]:


model.coef_


# In[369]:


X.head()


# 

# ## 축하드립니다. 여러분은 이제 로지스틱 모델을 구현하실 수 있게 되었습니다!
