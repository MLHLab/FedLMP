#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import io
import os
import collections


# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:





# In[ ]:


df = pd.read_csv(io.BytesIO(uploaded['InputNote8.csv']), encoding='unicode_escape')
print(df)


# In[ ]:


get_ipython().system('pip install sentence_transformers')
# For GPU
get_ipython().system('apt install libomp-dev')
get_ipython().system('pip install faiss-gpu')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sentence_transformers import SentenceTransformer
import faiss


# In[ ]:


df.shape


# In[ ]:


#Setup Train & Test
train_data, test_data = train_test_split(df, test_size=0.3, random_state=10)
print(train_data.shape)
print(test_data.shape)


# In[ ]:


print(train_data['disease_hierarch1'].value_counts())
print(test_data['disease_hierarch1'].value_counts())
print(train_data['disease_hierarch2'].value_counts())
print(test_data['disease_hierarch2'].value_counts())


# In[ ]:


model = SentenceTransformer("xlm-r-distilroberta-base-paraphrase-v1")

def text_to_vector(text):
    vector = model.encode(text)
    return vector


# In[ ]:


train_vectors = np.array([text_to_vector(text) for text in train_data['clinical_notes']])
test_vectors = np.array([text_to_vector(text) for text in test_data['clinical_notes']])

faiss.normalize_L2(train_vectors)
faiss.normalize_L2(test_vectors)


# In[ ]:


print("First value of normalized test vector:", test_vectors[0])
print("First value of normalized train vector:", train_vectors[0][0])


# In[ ]:


test_vectors.shape


# In[ ]:


index = faiss.IndexFlatL2(train_vectors.shape[1])
index.add(train_vectors)


# In[ ]:


k = 1  # Number of nearest neighbors to retrieve
distances, indices = index.search(test_vectors, k)


# In[ ]:


result_dft = pd.DataFrame(columns=['Test Index', 'actual_dis_hier1','predicted_dis_hier1', 'actual_dis_hier2','predicted_dis_hier2', 'Distance'])


# In[ ]:


# Predict XX for test data based on nearest neighbor with smallest distance
for i in range(len(test_data)):
    min_distance_index = indices[i][0]  # Index of nearest neighbor with smallest distance
    predicted_dis_hier1 = train_data.iloc[min_distance_index]['disease_hierarch1']
    predicted_dis_hier2 = train_data.iloc[min_distance_index]['disease_hierarch2']
    actual_dis_hier1 = test_data.iloc[i]['disease_hierarch1']
    actual_dis_hier2 = test_data.iloc[i]['disease_hierarch2']
    test_index = i
    distance = distances[i]
    result_dft = result_dft._append({'Test Index': test_index,
                                      'actual_dis_hier1': actual_dis_hier1,
                                      'predicted_dis_hier1': predicted_dis_hier1,
                                      'actual_dis_hier2': actual_dis_hier2,
                                      'predicted_dis_hier2': predicted_dis_hier2,
                                      'Distance': distance
                                    }, ignore_index=True)

display(result_dft)


# In[ ]:


def all_metrics(dft,actual, predicted, minor):
    y_true = dft[actual]
    y_pred = dft[predicted]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred,average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred,average='weighted', zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label=minor)
    f1 = f1_score(y_true, y_pred,average='weighted', zero_division=0)

    return {
        "Accuracy": round(accuracy,4),
        "Precision": round(precision,4),
        "Recall/ Sensetivity": round(recall,4),
        "Specificity": round(specificity,4),
        "f1" : round(f1,4)
    }

metrics1 = all_metrics(result_dft,'actual_dis_hier1', 'predicted_dis_hier1','NonViral')
print("Setup 1: Viral-NonViral")
display(metrics1)

metrics2 = all_metrics(result_dft,'actual_dis_hier2', 'predicted_dis_hier2','NonCovid')
print("Setup 2: Covid-NonCovid")
display(metrics2)


# In[ ]:




