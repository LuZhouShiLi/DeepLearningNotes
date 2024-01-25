#!/usr/bin/env python
# coding: utf-8

# In[2]:


from obspy import read


# In[3]:


file_dir = 'D:/02-SmartEarth/04-OriginalData/02-Hakuang_Exam_in_Hefei/02-Guowei/'
filename = '20230703_115900.sgy'


# In[4]:


st = read(file_dir + filename)


# In[7]:


tr = st[0]
print(tr.stats)


# In[10]:


st[0].plot()


# In[ ]:




