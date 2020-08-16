#!/usr/bin/env python
# coding: utf-8

# In[34]:


N = int(input())
rooms = [0] + [int(input()) for i in range(N)]


# In[38]:


result = []
for num in range(len(rooms)) :
    if num <= 1 : 
        result.append(rooms[num])
    elif num == 2 :
        result.append(rooms[1] + rooms[2])
    else :
        result.append(max(result[num-1], result[num-2]+rooms[num], result[num-3]+rooms[num-1]+rooms[num]))
print(result[-1])

