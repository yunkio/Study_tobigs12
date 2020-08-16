#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import multiprocessing
from gensim.models import Word2Vec, fasttext
from konlpy.tag import *


# ## 크롤링

# 네이버 뉴스 제목 크롤링하기

# In[2]:


# 네이버 뉴스 제목 크롤링 하기!
def crawler(maxpage,query,s_date,e_date):
    s_from = s_date.replace(".","")
    e_to = e_date.replace(".","")
    page = 1
    maxpage_t =(int(maxpage)-1)*10+1
    title_text = list()
    while page <= maxpage_t:
        url = "https://search.naver.com/search.naver?where=news&query=" + query + "&sort=0"+"&ds=" + s_date + "&de=" + e_date + "&nso=so%3Ar%2Cp%3Afrom" + s_from + "to" + e_to + "%2Ca%3A&start=" + str(page)   
        response = requests.get(url)
        html = response.text       
        #뷰티풀소프의 인자값 지정
        soup = BeautifulSoup(html, 'html.parser')        
        #태그에서 제목과 링크주소 추출
        atags = soup.select('._sp_each_title')
        for atag in atags:
            title_text.append(atag.text) #제목      
        #모든 리스트 딕셔너리형태로 저장
        result= {"title":title_text}  
        df = pd.DataFrame(result) #df로 변환
        page += 10
    return df
# 참고 : https://bumcrush.tistory.com/116


# In[3]:


df = crawler(300, "조국", "2019.09.04", "2019.09.08") # 조국에 대한 이슈를 크롤링! 기간은 내가 제주도 가있는 동안!


# ## 전처리

# In[4]:


text = df["title"]


# In[5]:


twitter = Okt()
def make_corpus_rm_stopwords(text):
    corpus = []
    for line in text:
        corpus.append(['/'.join(p) for p in twitter.pos(line) if p[1] not in ("Josa", "Punctuation", "Foreign", "Number")]) #전처리 
    return corpus

#주로 명사를 다룰 것이기 때문에 너무 세세하고 나누는 다른 방법보다 twitter가 더 유용할 것으로 보인다.


# In[6]:


corpus = make_corpus_rm_stopwords(text) # 말뭉치 형태로 변환


# ## 모델링

# ### 1. Skip Gram

# In[7]:


Skip_Gram_model = Word2Vec(corpus, size=3, window=3, min_count=4, workers=multiprocessing.cpu_count(), iter=10, sg=1)
# 3차원의 벡터로 변환, 주변 단어는 앞뒤로 세개까지, 코퍼스 내 출현 빈도 최소 4회, 스킵그램
# 표본이 그렇게 많지 않기 때문에 min_count도 크지 않은 4으로 설정, 문장이 길지 않은 뉴스 기사 제목 분석이기 때문에 window 역시 크지 않은 3로 설정했다.
# 역시 표본이 많지 않아 모델을 돌리는 속도가 크게 느리지 않기 때문에 iteration은 10 정도로 설정
# 차원은 3이면 충분해보임


# In[60]:


w2c = dict()
for item in Skip_Gram_model.wv.vocab :
    w2c[item] = Skip_Gram_model.wv.vocab[item].count


# In[61]:


words = Skip_Gram_model.wv.index2word
vectors = Skip_Gram_model.wv.vectors
Skip_Gram_model_result = dict(zip(words, vectors))
Skip_Gram_model.most_similar('사퇴/Noun', topn=20)


# In[62]:


w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2', 'x3'])
w2v_df['word'] = words
w2v_df = w2v_df[['word', 'x1', 'x2', 'x3']]


# ### 2. CBOW

# In[10]:


CBOW_model = Word2Vec(corpus, size=3, window=3, min_count=4, workers=multiprocessing.cpu_count(), iter=100, sg=0)


# In[11]:


CBOW_words = CBOW_model.wv.index2word
CBOW_vectors = CBOW_model.wv.vectors
CBOW_model_result = dict(zip(words, vectors))
CBOW_model.most_similar('사퇴/Noun', topn=20)


# In[12]:


CBOW_w2v_df = pd.DataFrame(CBOW_vectors, columns = ['x1', 'x2', 'x3'])
CBOW_w2v_df['word'] = CBOW_words
CBOW_w2v_df = w2v_df[['word', 'x1', 'x2', 'x3']]


# In[13]:


# CBOW 모델과 Skip-gram 모델을 비교하면 CBOW의 모델이 조금 더 논리적이라고 개인적으로 생각하지만, 실제 실험 결과에서는 Skip-gram이 CBOW에 비해 전체적으로 다소 좋은 결과를 내는 추세를 보인다. 현재는 대부분의 사람들이 Skip-gram 모델을 사용하는 것 같다.
# 출처 : https://shuuki4.wordpress.com/2016/01/27/word2vec-%EA%B4%80%EB%A0%A8-%EC%9D%B4%EB%A1%A0-%EC%A0%95%EB%A6%AC/
# 모델링 결과도 큰 차이가 없어보인다. Skip Gram 방법을 활용하여 분석해보기로 하자


# ## 인사이트 도출

# 어떤 단어가 어떤 단어들과 관련이 있는지를 클러스터링을 통해 분석한 후 각 클러스터를 살펴봐 인사이트를 도출하고자 한다.

# In[66]:


count = [Skip_Gram_model.wv.vocab[item].count for item in Skip_Gram_model.wv.vocab]
w2v_df["count"] = count # 데이터 프레임에 빈도수 추가


# In[67]:


from sklearn.cluster import KMeans


# In[91]:


data_points = w2v_df.iloc[:, 1:4].values
kmeans = KMeans(n_clusters=4).fit(data_points)
w2v_df['cluster_id'] = kmeans.labels_

# 약 5일간의 기간 동안의 데이터이므로 클러스터 갯수가 너무 많으면 제대로 주제로 나눠지기 힘들다고 판단해 4로 설정했다.


# In[92]:


w2v_df[w2v_df["cluster_id"] == 0].loc[:,("word", "count")].sort_values('count', ascending=False).head(15)


# In[93]:


w2v_df[w2v_df["cluster_id"] == 1].loc[:,("word", "count")].sort_values('count', ascending=False).head(15)


# In[94]:


w2v_df[w2v_df["cluster_id"] == 2].loc[:,("word", "count")].sort_values('count', ascending=False).head(15)


# In[95]:


w2v_df[w2v_df["cluster_id"] == 3].loc[:,("word", "count")].sort_values('count', ascending=False).head(15)


# 4개의 집단으로 클러스터링을 했으며, 나름대로 각각 주제를 잘 갖춰서 분류된거 같다.
# 
# 첫번째 집단의 경우 '임명' 과 '대통령', '문재인' 에 대한 이야기가 주로 언급되었다. 대통령이 임명을 감행함에 따라 혹은 여론조사를 통한 지지율 같은 주제로 볼 수 있었다.
# 
# 두번째 집단의 경우 '부인' 과 관련된 이슈인 것으로 보인다. 검색 결과 조국의 부인이 해명글을 올렸으며 포토라인 등에 대한 이야기가 나왔다.
# 
# 세번째 집단의 경우 '딸' 에 대한 이슈로 동양대 표창장 등의 이야기가 나왔다.
# 
# 네번째 집단의 경우 '주광덕' 의원이 청문회 과정에서 '조국' 에게 제기한 의혹들을 의미하며 아들 및 자녀들에 대한 이야기도 있으며 조국이 관련 인물과 통화를 했다는 의혹도 제기되어 '통화' 등의 키워드도 발견할 수 있었다.
# 
# 이와 같이 뉴스 기사 제목만을 클러스터링을 해 당시 기간동안 있었던 '조국' 에 관련된 여러 이슈들을 알아볼 수 있었다.

# In[ ]:




