# Che, Te veo
Aplicación de reconocimiento facial y detección de objecto en movimiento con utilización de librerías Python (Deepface y YOLO)

# Debes ejecutar en la carpeta donde se descargó o creaste
python app.py

# Requierimiento
* Tener Python instalado
  
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
* Descargar el archivo de YOLO python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
* python -m PyInstaller --noconfirm --clean --onefile --windowed --name CheTeVeo ^
  --add-data "yolov8n.pt;." ^
  --collect-all ultralytics ^
  --collect-all deepface ^
  --collect-all customtkinter ^
  --collect-all cv2 ^
  --collect-all gdown ^
  --copy-metadata gdown ^
  --copy-metadata deepface ^
  --copy-metadata retina-face ^
  --copy-metadata ultralytics ^
  --hidden-import gdown ^
  --hidden-import deepface ^
  app.py

 *Convertír a *.exe con sus librerías
