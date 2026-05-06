#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install faiss-cpu cryptography')


# In[ ]:


import faiss
from cryptography.fernet import Fernet


# In[ ]:


# Load existing FAISS index
index_path = "faiss.index"   # change if needed
index = faiss.read_index(index_path)

print("Loaded FAISS index")
print("Total vectors:", index.ntotal)

index_bytes = faiss.serialize_index(index)
print("Index serialized (bytes length):", len(index_bytes))

# Generate encryption key
key = Fernet.generate_key()

with open("secret.key", "wb") as f:
    f.write(key)

print("Encryption key saved: secret.key")

# Encrypt serialized index and save the same
cipher = Fernet(key)

encrypted_data = cipher.encrypt(index_bytes)

with open("faiss_encrypted.index", "wb") as f:
    f.write(encrypted_data)

print("Encrypted index saved: faiss_encrypted.index")

