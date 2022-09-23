
# **Tabla de contenido**
- [Requerimientos](#Requerimientos)
- [Preprocesamiento](#Preprocesamiento)
- [Algoritmos](#Algoritmos)
- [Bibliografía](#Bibliografía)

# **Requerimientos**

## **Librerías utilizadas**

```
pandas
matplotlib
numpy
faiss
rtree
sklearn
```

## **Instalación de librería `faiss`**

- Primero, instalar **Miniconda3**
- Luego abres una terminal de anaconda (buscalo como Anaconda Powershell, se va a instalar junto al Miniconda), y tienes dos alternativas:
 ejecutas la lista de comandos de abajo, o ejecutas esta linea de codigo ubicado en el directorio principal del proyecto: 
	Si quieres ejecutar el proyecto con lo que se ha estado corriendo hasta ahora: conda env create --file environment.yml
	(Aun no probar) En cambio, aun falta probar si los comandos con gpu funcionan, si quieres ejecutarlo seria: conda env create --file environmentgpu.yml
```
conda update --all
conda create -n openday	
conda activate openday

conda install -c anaconda flask
conda install -c conda-forge faiss-cpu
conda install -c conda-forge matplotlib
conda install -c conda-forge pandas
conda install -c conda-forge numpy
conda install -c conda-forge rtree
conda install -c conda-forge face_recognition

```
- **Para ejecutar la aplicación**
```
-cambia de directorio a donde sea que tengas la carpeta con este README
-en la consola pondrias: cd TuDirectorio
-pones python app.py
```