import faiss
import numpy as np
import json
encoded_faces = {} 
with open('encoded_data.json', 'r') as f:
  encoded_faces = json.load(f)

vector = []
faces= []
for key in encoded_faces: 
  vector.append(np.array(encoded_faces[key], dtype='f'))
  faces.append(key)
vector = np.array(vector, dtype='f')

M = 32
ef_search = 8
ef_construction = 64
d = 128
k = 8

index  = faiss.IndexHNSWFlat(d,M)
index.hnsw.efConstruction = 40
index.hnsw.efSearch = ef_search


index.add(vector)

faiss.write_index(index, "prepared_data.index")