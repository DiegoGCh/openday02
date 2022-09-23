import faiss
import json
import numpy as np
import time

def faiss_process():
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
    query = np.reshape(np.array(vector[0],dtype='f'), (1,128))
    print(query.shape)
    index  = faiss.IndexHNSWFlat(d,M)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = ef_search

    index.add(vector)
    n, indx= index.search(query, k=11)
    indx = indx[0].astype(int)
    result = []
    for i in indx[1:]:
        print(faces[i])
        result.append(faces[i])

    return indx, result, faces
