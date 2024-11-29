# DETECCIÓN DE ENFERMEDADES EN CULTIVOS DE TOMATE CHONTO A TRAVÉS DE ANÁLISIS COMPUTACIONAL DE SUS HOJAS A PARTIR DE VISIÓN ARTIFICIAL.

## Lenguaje de programación
- Versión de python recomendada 3.8, 3.9

## Librerias utilizadas
- OpenCv
- Numpy
- PyTorch
- ultralytics
- matplotlib
- dotenv
- twilio
- flask

## Instalación

### 1. Configuracion del repositorio de GitHub:
1.1. Clonar el repositorio:
   ``` git clone https://github.com/BryanViancha/tomato_disease_detected.git```

1.2. Acceder al repositorio en el IDE ```tomato_disease_detected```

### 2. Instalacion y configuracion de entorno de Python:
2.1. Primero descargar e instalar Python version recomendada

2.2. Crear un entorno de desarollo con python para instalar las dependencias 
``` python -m venv nombre_del_entorno```

2.3. Activar entorno creado
``` venv\Scripts\activate```

2.4. Despues de activado el entorno para instalar todos los paquetes y librerias del proyecto, ejecutar el siguente comando
   ```pip install -r requirements.txt```

2.5. Creacion de archivo requirements, listado de las las bibliotecas y paquetes usados
```pip freeze > requirements.txt```

### 3. Configuracion e instalacion de YOLOv8
3.1. En la raiz del proyecto instalar YOLOv8 ```pip install ultralytics```

### Entrenamiento del modelo:
3.2. Crear carpetas de esta forma: carpeta train y val

3.3. Dentro de cada una deben de haber 2 carpetas que son las imagenes y las etiquetas

3.4. Ingresar las imagenes y las etiquetas en cada carpeta en este caso train = 70% de imagenes y etiquetas y val = 30%

3.5. Crear el archivo __init__.py en cada carpeta que sea requerida

3.6. Crear un archivo llamado data.yaml con la siguiente estructura:

   ruta de las imagenes

   train: data/images/train 

   val: data/images/val

   nc: 2   # Número de clases

   names: [ 'minador', 'alternaria' ]

3.7. Para el entrenamiento del modelo ejecutar el siguiente comando en la terminal: ```python train.py --img 640 --batch 16 --epochs 50 --data data.yaml  --weights yolov8m.pt```

#### Instalacion de las principales librerias
- Instalacion de ultralytics ```pip install ultralytics```

- Instalacion opencv ```pip install opencv-python```

- Instalacion numpy ```pip install numpy```

- Instalacion matplotlib ```pip install matplotlib```

- Instalacion Flask ```pip install flask```

- Instalacion dotenv ```pip install python-dotenv```

- Instalacion twilio ```pip install twilio```

- Instalacion torch ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu```

### 4. Ejecutar el archivo principal para correr el proyecto de deteccion
- Antes de ejecutar el app.py, dirigirse al archivo .ENV donde estan las variables de entorno y remplazar:

- TWILIO_ACCOUNT_SID por su token generado en twilio

- TWILIO_AUTH_TOKEN por su token generado en twilio

- TARGET_PHONE_NUMBER por su numero de telefono enlazado desde twilio a donde le llegaran los mensajes

- IP_CAMERA por la IP de su camara

- CAMERA_USER por la usuario de su camara

- CAMERA_PASSWORD por la clave de su camara

- El script principal del proyecto es app.py, para ejecutar el codigo en la carpeta app esta el script, ejecutar el siguiente comando ```python app.py```



