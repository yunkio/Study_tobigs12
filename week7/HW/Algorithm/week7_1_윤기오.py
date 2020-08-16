#!/usr/bin/env python
# coding: utf-8

# In[15]:


from itertools import combinations


# In[13]:


N,X = map(int, input().split(" "))
nums = list(map(int, input().split(" ")))


# In[16]:


sums = []
for i in range(N+1) :
    for subset in combinations(nums, i):
        sums.append(sum(subset))
results = [num - X for num in sums if num - X <= 0]
print(max(results) + X)

