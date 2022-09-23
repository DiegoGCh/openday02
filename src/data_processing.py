import glob
import face_recognition
import numpy as np
import pandas as pd

path = r'lfw3/**/*.jpg'
files = glob.glob(path, recursive=True)
import json
encoded_faces = { }
for file in files: 
    try:
        encoded_faces[file]= (face_recognition.face_encodings(face_recognition.load_image_file( file ))[0]).tolist()        
    except:
        continue

jso = json.dumps(encoded_faces) 
js =  open('encoded_data.json','w')
js.write(jso)
js.close()
