#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install faiss-cpu cryptography')


# In[ ]:


import faiss
from cryptography.fernet import Fernet


# In[ ]:


# Load same encryption key
with open("secret.key", "rb") as key_file:
    key = key_file.read()

cipher = Fernet(key)

# Read encrypted FAISS file
with open("faiss_encrypted.index", "rb") as f:
    encrypted_data = f.read()

# Decrypt the file
decrypted_data = cipher.decrypt(encrypted_data)

with open("faiss_decrypted.index", "wb") as f:
    f.write(decrypted_data)

print("FAISS index decrypted successfully")

# Load FAISS index

index = faiss.read_index("faiss_decrypted.index")

print("FAISS index loaded at server")
print("Total vectors in index:", index.ntotal)

