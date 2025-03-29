#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install faiss-gpu')
get_ipython().system("pip install llama-index 'google-generativeai>=0.3.0'")
get_ipython().system("pip install llama-index 'google-generativeai'")


# In[4]:


import numpy as np
import pandas as pd
import io
import os
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import faiss
from llama_index.embeddings.gemini import GeminiEmbedding


# In[ ]:


GOOGLE_API_KEY = "AIzaSyARHVhoxn_XH3Cdq8mnYOEM8xW9pnJ3xdI"  # GOOGLE API key is set here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


train_data_asia = pd.read_csv(io.BytesIO(uploaded['train_data_asia.csv']), encoding='unicode_escape')
print(df)


# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


test_data_asia = pd.read_csv(io.BytesIO(uploaded['test_data_asia.csv']), encoding='unicode_escape')
print(df)


# In[ ]:


print(train_data_asia['disease_hierarch1'].value_counts())
print(test_data_asia['disease_hierarch1'].value_counts())
print(train_data_asia['disease_hierarch2'].value_counts())
print(test_data_asia['disease_hierarch2'].value_counts())


# In[ ]:


#Setup Train & Test
train_data=train_data_asia
test_data=test_data_asia


# In[ ]:


# Declare the model is use and define the embedding function
model_name = "models/text-embedding-004"

embed_model = GeminiEmbedding(
    model_name=model_name, api_key=GOOGLE_API_KEY, title="this is a document"
)


# In[ ]:


train_vectors_1 = np.array([embed_model.get_text_embedding(text) for text in train_data['clinical_notes']])
test_vectors_1 = np.array([embed_model.get_text_embedding(text) for text in test_data['clinical_notes']])


# In[ ]:


train_vectors = train_vectors_1.astype(np.float32)
test_vectors = test_vectors_1.astype(np.float32)


# In[ ]:


faiss.normalize_L2(train_vectors)
faiss.normalize_L2(test_vectors)


# In[ ]:


print("First value of normalized test vector:", test_vectors[0])
print("First value of normalized train vector:", train_vectors[0][0])


# In[ ]:


print(train_vectors.shape)
print(test_vectors.shape)


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


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# Save the FAISS index
index_file_path = '/content/drive/MyDrive/Vector/asia_vector.index'
faiss.write_index(index, index_file_path)


# In[ ]:





# In[ ]:


# Load and run with Central for asia Data
index = faiss.read_index('/content/drive/MyDrive/central_vector.index')


# In[ ]:


csv_file_path = '/content/drive/MyDrive/Vector/central_class.csv'
central_class = pd.read_csv(csv_file_path)
print(central_class)


# In[ ]:


k = 1  # Number of nearest neighbors to retrieve
distances, indices = index.search(test_vectors, k)


# In[ ]:


result_dft = pd.DataFrame(columns=['Test Index', 'actual_dis_hier1','predicted_dis_hier1', 'actual_dis_hier2','predicted_dis_hier2', 'Distance'])


# In[ ]:


# Predict XX for test data based on nearest neighbor with smallest distance
for i in range(len(test_data)):
    min_distance_index = indices[i][0]  # Index of nearest neighbor with smallest distance
    predicted_dis_hier1 = central_class.iloc[min_distance_index]['disease_hierarch1']
    predicted_dis_hier2 = central_class.iloc[min_distance_index]['disease_hierarch2']
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





# In[ ]:


# Load and run with Europe for Asia Data
index = faiss.read_index('/content/drive/MyDrive/europe_vector.index')


# In[ ]:


csv_file_path = '/content/drive/MyDrive/Vector/europe_class.csv'
europe_class = pd.read_csv(csv_file_path)
print(europe_class)


# In[ ]:


k = 1  # Number of nearest neighbors to retrieve
distances, indices = index.search(test_vectors, k)


# In[ ]:


result_dft = pd.DataFrame(columns=['Test Index', 'actual_dis_hier1','predicted_dis_hier1', 'actual_dis_hier2','predicted_dis_hier2', 'Distance'])


# In[ ]:


# Predict XX for test data based on nearest neighbor with smallest distance
for i in range(len(test_data)):
    min_distance_index = indices[i][0]  # Index of nearest neighbor with smallest distance
    predicted_dis_hier1 = europe_class.iloc[min_distance_index]['disease_hierarch1']
    predicted_dis_hier2 = europe_class.iloc[min_distance_index]['disease_hierarch2']
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





# In[ ]:


# Load and run with America for Asia Data
index = faiss.read_index('/content/drive/MyDrive/america_vector.index')


# In[ ]:


csv_file_path = '/content/drive/MyDrive/Vector/america_class.csv'
america_class = pd.read_csv(csv_file_path)
print(america_class)


# In[ ]:


k = 1  # Number of nearest neighbors to retrieve
distances, indices = index.search(test_vectors, k)


# In[ ]:


result_dft = pd.DataFrame(columns=['Test Index', 'actual_dis_hier1','predicted_dis_hier1', 'actual_dis_hier2','predicted_dis_hier2', 'Distance'])


# In[ ]:


# Predict XX for test data based on nearest neighbor with smallest distance
for i in range(len(test_data)):
    min_distance_index = indices[i][0]  # Index of nearest neighbor with smallest distance
    predicted_dis_hier1 = america_class.iloc[min_distance_index]['disease_hierarch1']
    predicted_dis_hier2 = america_class.iloc[min_distance_index]['disease_hierarch2']
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





# In[ ]:





# In[ ]:




