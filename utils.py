import faiss
import glob
import hashlib
import heapq as hq
import json
import numpy as np
import os

import rtree

m = hashlib.md5()
hash_to_path = {}

# FAISS CPU HNSW
def faiss_search(data,query):    
    faces= []
    for key in data:     
        faces.append(key)


    query = np.reshape(np.array(query,dtype='f'), (1,128))
    index  = faiss.read_index("prepared_data.index")

    n, indx= index.search(query, k=3)
    indx = indx[0].astype(int)
    result = []
    for i in indx:
        face = faces[i]
        result.append(face.replace('\\','/'))

    return indx, result, faces
# FAISS GPU FLAT
def faiss_search_gpu(data, query):    
    faces= []
    for key in data:         
        faces.append(key)

    query = np.reshape(np.array(query,dtype='f'), (1,128))

    index_flat = faiss.read_index("prepared_data_for_gpu.index")
    

    res = faiss.StandardGpuResources()  # use a single GPU
    gpu_index  = faiss.index_cpu_to_gpu(res, 0, index_flat)
    

    n, indx= gpu_index.search(query, k=8)
    indx = indx[0].astype(int)
    result = []
    for i in indx:
        result.append(faces[i])

    return indx, result, faces
# FAISS IVF GPU
def faiss_search_ivf(data, query):    
    faces= []
    for key in data:         
        faces.append(key)

    query = np.reshape(np.array(query,dtype='f'), (1,128))

    index_ivf = faiss.read_index("prepared_data_ivf_for_gpu.index")


    res = faiss.StandardGpuResources()  # use a single GPU
    gpu_index  = faiss.index_cpu_to_gpu(res, 0, index_ivf)  
    

    n, indx= gpu_index.search(query, k=8)
    indx = indx[0].astype(int)
    result = []
    for i in indx:
        result.append(faces[i])

    return indx, result, faces


# KNN + RANGE SEARCH
def knn_search(Q, k, data):
    result = []
    for idx in data:
        try:             
            dist = np.linalg.norm(np.array(data[idx]) - np.array(Q))            
            hq.heappush( result, (dist,idx) )        
        except:
            continue
    return hq.nsmallest(k, result)

def knn_range_search(Q, r, data):
    result = []
    for idx in data:   
        if idx == Q: 
            continue
        try:
            dist = np.linalg.norm(np.array(data[idx]) - np.array(data[Q]))            
            if  dist < r:
                result.append((dist, idx))
        except:
            continue
    return result

# RTREE
def rtree_search(Q, k, ind):
    ids_ = ind.nearest(Q, num_results=k)
    return [hash_to_path[str(hsh)] for hsh in ids_]

# UTILS
def get_id(some_str):
    some_str = some_str.encode('utf-8')
    m.update(some_str)
    h = str(int(m.hexdigest(), 16))[0:12]
    return int(h)


'''
def get_metadata():
    path = r'static/lfw/**/*.jpg'
    files = [p.replace('static/', '') for p in glob.glob(path, recursive=True)]
    encoded_faces = {}
    with open('./encoded_data.json', 'r') as f:
        encoded_faces = json.load(f)

    prop = rtree.index.Property()
    prop.dimension = 128
    prop.buffering_capacity = 8
    ind = rtree.index.Index("feature_vector", properties=prop)

    global hash_to_path
    with open('rtree_ids.json', 'r') as f:
        hash_to_path = json.load(f)
    
    return files, encoded_faces, ind

'''
