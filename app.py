import random

import os
import json
from time import time
from flask import Flask, flash, redirect, render_template,request, url_for
from utils import faiss_search
from werkzeug.utils import secure_filename


import base64

import face_recognition
import time
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './static/img/'
app.secret_key = "secret key"

K = 8
'''
FILES, ENCODED_FACES, IND_RTREE = get_metadata()
'''

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/KNNHNSW', methods=['GET', 'POST'])
def KNN_hnsw():
    imagenes = None

    if request.method == 'GET':
        return render_template('knnhnsw.html')

    if request.method == 'POST':
        nombre = request.form['name']
        #print(nombre)
        fileURI = request.form['fileURL']
        #print(fileURI)
        
        #separa lo innecesario
        head, data = fileURI.split(',', 1)

        #agarra la extension
        file_ext = head.split(';')[0].split('/')[1]

        #Decodificar
        data_img = base64.b64decode(data)

        # escribir img a file
        filename = secure_filename(nombre)
        print(filename)
        print(filename+file_ext)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(filepath)

        with open(filepath + '.' + file_ext, 'wb') as f:
            f.write(data_img)
        
        file_dir = app.config['UPLOAD_FOLDER']+filename+"."+file_ext #.static/ son 9caracteres
        q_image = (face_recognition.face_encodings(face_recognition.load_image_file( file_dir ))[0])        
        query_image = q_image 

        encoded_faces = {}
        with open('./encoded_data.json', 'r') as f:
            encoded_faces = json.load(f)
        ENCODED_FACES = encoded_faces
    
        inicio = time.process_time()
        indx, result, faces =  faiss_search(ENCODED_FACES, query_image)
        fin =  time.process_time() - inicio
        print(result)
        
        return render_template('knnhnsw.html', imagenes=result, imagenprop = file_dir[9:], tiempo=fin)
        

    
# Run Script
if __name__ == "__main__":
    app.run(debug=True, port=5000)
