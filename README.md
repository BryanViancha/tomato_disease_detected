# Detección y Clasificación de Enfermedades en el Tomate

Este proyecto utiliza técnicas de Deep Learning para detectar y clasificar enfermedades en hojas de tomate.

## Paquetes utilizados

- Python 3.8
- Anaconda
- Keras
- TensorFlow
- OpenCv
- Numpy
- PyTorch
- ultralytics
- matplotlib


## Instalación

### 1. Instalacion y configuracion de Anaconda:
- Primero descargar e instalar Anaconda
- Abrir la terminal de Anaconda Prompt como administrador
- Ejecutar el siguiente comando para verificar la version de anaconda instalada:
   - conda --version
- Crear entorno de desarrollo con conda
``` conda create -n tomato_disease_detected python=3.8```

### 2. Configuracion del repositorio de GitHub:
2. Clonar el repositorio:
   ``` git clone https://github.com/BryanViancha/tomato_disease_detected.git```
3. Acceder al repositorio en el IDE ```tomato_disease_detected```

### 3. Configuracion del proyecto en Pycharm
4. Abrir el proyecto clonado en Pycharm
5. Ir a settings/Project/ python interpreter, seleccionar el entorno creado en la carpeta de envs de Anaconda, o Anaconda3, o Conda y seleccionar (python.exe)
6. Aplicar cambios y activar el entorno 
```conda activate tomato_disease_detected```
#### Ejemplo: (tomato_disease_detected) PS C:\Users\Brayan Viancha\Documents\GitHub\tomato_disease_detected> 
7. Despues de activado el entorno para instalar todos los paquetes y librerias del proyecto, correr el siguente comando
   ```pip install -r requirements.txt```
8. Creacion de archivo requirements, listado de las las bibliotecas y paquetes usados
```pip freeze > requirements.txt```

### 4. Configuracion e instalacion de YoloV5
9. En la raiz del proyecto y clonar el repositorio de YoloV5 ```git clone https://github.com/ultralytics/yolov5.git```

#### Entrenamiento del modelo:
10. En la carpeta data/images crear las carpetas de train y val
11. En la carpeta data crear la carpeta labels y dentro crear las subcarpetas de train y val
12. Ingresar las imagenes y las etiquetas en cada carpeta en este caso train = 70% de imagenes y etiquetas y val = 30% de imagenes y etiquetas
13. Crear el archivo __init__.py en cada carpeta que sea requerida
14. Crear el archivo en la carpeta data llamado data.yaml con la siguiente estructura:

ruta de las imagenes
train: data/images/train 
val: data/images/val

nc: 3  # Número de clases
names: [ 'minador', 'alternaria', 'hojaSana' ]

15. Entrar a la carpeta de YoloV5 ```cd yolov5```
16. Ejecutar el modelo de procesamiento de imagenes: ```python train.py --img 640 --batch 16 --epochs 100 --data data/data.yaml --weights yolov5s.pt```

#### Librerias y dependencias
Instalacion de Pytorch ```pip install torch torchvision torchaudio```

Instalacion Pyyaml ```pip install pyyaml```

Instalacion tqdm ```pip install tqdm```

Instalacion Pandas ```pip install pandas```

Instalacion Flask ```pip install flask```

