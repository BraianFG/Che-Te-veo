# Che, Te veo
Aplicación de reconocimiento facial y detección de objecto en movimiento con utilización de librerías Python (Deepface y YOLO)

# Debes ejecutar en la carpeta donde se descargó o creaste
python app.py

# Librerías a instalar previamente
* pip install customtkinter
* pip install deepface
* pip install ultralytics
* pip install opencv-python
* pip install Pillow
* pip install matplotlib 
* pip install numpy
* pip install onvif-zeep
  
# Sonido 
* Para esto se utiliza FFMEG , se debe descargar https://ffmpeg.org/download.html siguiendo las instrucciones del sitio web

# Convertir a un archivo .exe
Hay que instalar Pyinstaller

* pip install pyisntaller
* python -m PyInstaller --clean --noconsole --onedir --add-data "yolov8n.pt;." --collect-all gdown --collect-all deepface --name="CheTeVeo" app.py
