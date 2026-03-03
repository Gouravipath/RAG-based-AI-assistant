import requests
import os
import json
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

def create_embedding(text):
    r = requests.post("http://localhost:11434/api/embed",
                  json={
        "model": "bge-m3",
        "input": text
    })
    embedding = r.json()['embeddings']
    return embedding

# a = create_embedding(["cats", "Hay"])
# print(a)


jsons = os.listdir("json")
my_dict = []
chunk_id = 0
for json_file in jsons:
    with open(f'json/{json_file}') as f:
     content = json.load(f)
     print(f'Creating embedding for {json_file}')
     embedding = create_embedding([c['Text'] for c in content['Chunks']])
    for i , chunk in enumerate(content['Chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embedding[i]
        chunk_id += 1
        my_dict.append(chunk)
    
            

    
df = pd.DataFrame.from_records(my_dict)
# Save this dataframe
joblib.dump(df, 'embeddings.joblib')