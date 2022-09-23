import faiss
import numpy as np
import json
import time

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
index  = faiss.read_index("prepared_data.index")



n, indx= index.search(query, k=11)
indx = indx[0].astype(int)
print(n)
result = []
for i in indx[1:]:
  print(faces[i])
  result.append(faces[i])


import PIL
import matplotlib.pyplot as plt


def show_similar(query_image, neighbors):
    axes = min(len(neighbors) + 1, 5)       # max_axes
    fig, ax = plt.subplots(1, axes, figsize=(20,20))

    im = plt.imread(query_image)
    ax[0].imshow(im, extent=[0, 100, 0, 100])
    ax[0].axis('off')
    ax[0].set_title('IMAGEN DE CONSULTA')

    for i in range(1, axes):
        im_path = neighbors[i-1]
        im = plt.imread(im_path)
        ax[i].imshow(im,  extent=[0, 100, 0, 100])
        ax[i].axis('off')
        ax[i].set_title(f"IMAGEN SIMILAR #{i}")

    plt.show()

show_similar(faces[indx[0]],result)



