#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install faiss-gpu')


# In[4]:


import numpy as np
import pandas as pd
import io
import os
import collections
import faiss


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# Load Europe
index_europe = faiss.read_index('/content/drive/MyDrive/europe_vector.index')


# In[ ]:


# Load Asia
index_asia = faiss.read_index('/content/drive/MyDrive/asia_vector.index')


# In[ ]:


# Load America
index_america = faiss.read_index('/content/drive/MyDrive/america_vector.index')


# In[ ]:


# Central Index
index_central = faiss.IndexReplicas()
index_central.add_index(index_europe)
index_central.add_index(index_asia)
index_central.add_index(index_america)


# In[ ]:


# Save the FAISS index
index_file_path = '/content/drive/MyDrive/Vector/central_vector.index'
faiss.write_index(index, index_file_path)


# In[ ]:




