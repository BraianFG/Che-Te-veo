# app.py - DeepFace + YOLO con Estadísticas y Análisis Predictivo
import customtkinter as ctk
from collections import Counter, defaultdict, deque
from tkinter import filedialog, messagebox
from deepface import DeepFace
from ultralytics import YOLO
from PIL import Image
import cv2
import os
import threading
import time
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np 
import subprocess

# Optimización RTSP global
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Configuración de matplotlib para tema oscuro
plt.style.use("dark_background")


class DeepFaceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Che ,Te Veo")
        self.geometry("1480x920")
        self.minsize(1300, 820)

        self.img1_path = None
        self.img2_path = None

        # Audio RTSP
        self.audio_process = None
        self.grabando_audio = False
        self.ultimo_audio_guardado = 0.0
        self.intervalo_audio = 10.0
        
        # Variables de cámara
        self.cap = None
        self.camera_running = False
        self.frame_count = 0
        self.current_faces = []
        self.current_objects = []
        self.current_alturas = []

        # Control de frames por hilo independiente
        self.latest_raw_frame = None
        self.frame_lock = threading.Lock()

        # Control de eventos
        self.ultimos_objetos = []
        self.ultimo_guardado = 0.0
        self.intervalo_registro = 2.0
        self.intervalo_registro_persona = 8.0
        self.ultimo_registro_persona = 0.0
        self.es_webcam = True
        self.zoom_factor = 1.0

        # Filtros de detección
        self.person_conf_min = 0.50
        self.person_aspect_min = 0.50
        self.person_min_height_px = 0.50

        # Persistencia y control de movimiento
        self.persistent_boxes = deque(maxlen=4)
        self.persistent_objects = []
        self.persistent_alturas = []

        self.object_positions = {}
        self.static_threshold = 40
        self.movement_threshold = 80

        self.current_frame_for_analysis = None
        self.analysis_lock = threading.Lock()

        self.carpeta_registros = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registros")
        os.makedirs(self.carpeta_registros, exist_ok=True)

        # Modelo YOLO
        try:
            self.yolo_model = YOLO("yolov8n.pt")
        except Exception:
            try:
                self.yolo_model = YOLO("yolov8s.pt")
            except Exception as e2:
                messagebox.showerror("Error", f"No se pudo cargar YOLO:\n{e2}")
                self.yolo_model = None

        # ====================== INTERFAZ ======================
        self.label_titulo = ctk.CTkLabel(
            self, text="🎭 Che , Te Veo",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.label_titulo.pack(pady=(8, 4))

        # Pestañas principales
        self.tabview = ctk.CTkTabview(self, width=1420, height=820)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=4)

        self.tab_vivo = self.tabview.add("📹 En Vivo")
        self.tab_stats = self.tabview.add("📊 Estadísticas")
        self.tab_historial = self.tabview.add("📋 Historial")

        # ---------- PESTAÑA EN VIVO ----------
        self._construir_tab_vivo()

        # ---------- PESTAÑA ESTADÍSTICAS ----------
        self._construir_tab_estadisticas()

        # ---------- PESTAÑA HISTORIAL ----------
        self._construir_tab_historial()

        self.status = ctk.CTkLabel(self, text="Listo", font=ctk.CTkFont(size=12), text_color="#aaaaaa")
        self.status.pack(pady=(2, 6))

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.cambiar_tipo_camara("Cámara Local (Webcam)")

    # ==================================================================
    # CONSTRUCCIÓN DE PESTAÑAS
    # ==================================================================
    def _construir_tab_vivo(self):
        main_frame = ctk.CTkFrame(self.tab_vivo)
        main_frame.pack(fill="both", expand=True, padx=6, pady=6)

        # IZQUIERDA
        left = ctk.CTkScrollableFrame(main_frame, width=320)
        left.pack(side="left", fill="y", padx=(4, 4), pady=4)

        ctk.CTkLabel(left, text="Fuente de Cámara", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(8, 4))
        self.tipo_camara = ctk.CTkComboBox(
            left, values=["Cámara Local (Webcam)", "Cámara de Seguridad (RTSP)"],
            width=280, command=self.cambiar_tipo_camara
        )
        self.tipo_camara.set("Cámara Local (Webcam)")
        self.tipo_camara.pack(pady=3)

        self.label_rtsp = ctk.CTkLabel(left, text="URL RTSP:", font=ctk.CTkFont(size=12))
        self.entry_rtsp = ctk.CTkEntry(left, width=290, placeholder_text="rtsp://admin:clave@192.168.1.105:554/onvif1")

        self.btn_camara = ctk.CTkButton(
            left, text="▶️ Iniciar Cámara", command=self.toggle_camara,
            fg_color="#e67e22", hover_color="#d35400", width=280, height=38,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.btn_camara.pack(pady=(10, 8))

        ctk.CTkLabel(left, text="Modelo Facial", font=ctk.CTkFont(size=13)).pack(pady=(6, 2))
        self.modelo = ctk.CTkComboBox(left, values=["VGG-Face", "Facenet", "ArcFace", "SFace"], width=220)
        self.modelo.set("VGG-Face")
        self.modelo.pack(pady=2)

        ctk.CTkLabel(left, text="Calibración altura (cm/píxel)", font=ctk.CTkFont(size=12)).pack(pady=(8, 2))
        self.entry_factor_altura = ctk.CTkEntry(left, width=220, placeholder_text="Vacío = 0.45")
        self.entry_factor_altura.pack(pady=2)

        self.sonido_activo = ctk.BooleanVar(value=True)
        self.switch_sonido = ctk.CTkSwitch(left, text="🔔 Sonido al registrar", variable=self.sonido_activo)
        self.switch_sonido.pack(pady=(12, 6), anchor="w", padx=10)

        ctk.CTkLabel(left, text="Imágenes de referencia", font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(14, 4))
        self.btn_img1 = ctk.CTkButton(left, text="📁 Cargar Imagen 1", command=self.cargar_img1, width=220)
        self.btn_img1.pack(pady=2)
        self.btn_img2 = ctk.CTkButton(left, text="📁 Cargar Imagen 2", command=self.cargar_img2, width=220)
        self.btn_img2.pack(pady=2)
        self.btn_verificar = ctk.CTkButton(left, text="🔍 Verificar Imágenes", command=self.iniciar_verificacion,
                                           fg_color="#27ae60", width=220)
        self.btn_verificar.pack(pady=(6, 4))

        # CENTRO
        center = ctk.CTkFrame(main_frame)
        center.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        header = ctk.CTkFrame(center, fg_color="transparent")
        header.pack(fill="x", padx=6, pady=2)
        ctk.CTkLabel(header, text="Vista en Vivo", font=ctk.CTkFont(size=14, weight="bold")).pack(side="left")

        zoom_frame = ctk.CTkFrame(header, fg_color="transparent")
        zoom_frame.pack(side="right")
        ctk.CTkButton(zoom_frame, text="-", width=28, height=24, command=self.zoom_out).pack(side="left", padx=2)
        self.zoom_label = ctk.CTkLabel(zoom_frame, text="1.0x", width=40)
        self.zoom_label.pack(side="left")
        ctk.CTkButton(zoom_frame, text="+", width=28, height=24, command=self.zoom_in).pack(side="left", padx=2)

        self.camera_label = ctk.CTkLabel(center, text="Cámara apagada", width=640, height=400,
                                         fg_color="#1a1a1a", corner_radius=8)
        self.camera_label.pack(pady=4, padx=6)

        self.zoom_slider = ctk.CTkSlider(center, from_=1.0, to=3.0, number_of_steps=20,
                                         command=self.cambiar_zoom_slider, width=600)
        self.zoom_slider.set(1.0)
        self.zoom_slider.pack(pady=2)

        self.live_result = ctk.CTkTextbox(center, height=90, font=ctk.CTkFont(size=12))
        self.live_result.pack(pady=4, padx=8, fill="x")
        self.live_result.insert("1.0", "Detecciones en vivo aparecerán aquí...")

        # DERECHA (registro rápido)
        right = ctk.CTkFrame(main_frame, width=280)
        right.pack(side="right", fill="y", padx=(4, 4), pady=4)
        right.pack_propagate(False)

        ctk.CTkLabel(right, text="📋 Últimos eventos", font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(8, 4))
        self.resultado = ctk.CTkTextbox(right, font=ctk.CTkFont(size=11))
        self.resultado.pack(pady=4, padx=6, fill="both", expand=True)
        self.resultado.insert("1.0", "Aquí se muestran los eventos recientes...\n")

        self.btn_limpiar = ctk.CTkButton(right, text="🗑️ Limpiar", command=self.limpiar_registro,
                                         fg_color="#7f8c8d", width=200)
        self.btn_limpiar.pack(pady=6)

    def _construir_tab_estadisticas(self):
        frame = ctk.CTkFrame(self.tab_stats)
        frame.pack(fill="both", expand=True, padx=8, pady=8)

        top = ctk.CTkFrame(frame, fg_color="transparent")
        top.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(top, text="Análisis Estadístico y Predictivo",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(side="left", padx=6)

        self.btn_actualizar_stats = ctk.CTkButton(
            top, text="🔄 Actualizar estadísticas", command=self.actualizar_estadisticas,
            fg_color="#2980b9", width=200
        )
        self.btn_actualizar_stats.pack(side="right", padx=6)

        content = ctk.CTkFrame(frame)
        content.pack(fill="both", expand=True)

        left_stats = ctk.CTkScrollableFrame(content, width=340)
        left_stats.pack(side="left", fill="y", padx=(4, 6), pady=4)

        ctk.CTkLabel(left_stats, text="📈 Resumen del período",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(8, 6))

        self.label_periodo = ctk.CTkLabel(left_stats, text="Período: últimos 7 días", text_color="#aaaaaa")
        self.label_periodo.pack(pady=2)

        self.metricas_frame = ctk.CTkFrame(left_stats, fg_color="#1e1e1e")
        self.metricas_frame.pack(fill="x", padx=4, pady=6)

        self.lbl_total_personas = ctk.CTkLabel(self.metricas_frame, text="👤 Personas: —", anchor="w")
        self.lbl_total_personas.pack(fill="x", padx=10, pady=3)
        self.lbl_total_vehiculos = ctk.CTkLabel(self.metricas_frame, text="🚗 Vehículos: —", anchor="w")
        self.lbl_total_vehiculos.pack(fill="x", padx=10, pady=3)
        self.lbl_total_animales = ctk.CTkLabel(self.metricas_frame, text="🐕 Animales: —", anchor="w")
        self.lbl_total_animales.pack(fill="x", padx=10, pady=3)
        self.lbl_total_eventos = ctk.CTkLabel(self.metricas_frame, text="📋 Total eventos: —", anchor="w")
        self.lbl_total_eventos.pack(fill="x", padx=10, pady=3)

        ctk.CTkLabel(left_stats, text="👥 Distribución de género",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(14, 4))
        self.lbl_genero = ctk.CTkLabel(left_stats, text="Sin datos aún", justify="left")
        self.lbl_genero.pack(anchor="w", padx=8)

        ctk.CTkLabel(left_stats, text="🎂 Edad promedio",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(12, 4))
        self.lbl_edad = ctk.CTkLabel(left_stats, text="Sin datos aún")
        self.lbl_edad.pack(anchor="w", padx=8)

        ctk.CTkLabel(left_stats, text="🔮 Predicción (próximas horas)",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(16, 4))
        self.lbl_prediccion = ctk.CTkLabel(left_stats, text="Actualizá las estadísticas para ver la predicción",
                                           justify="left", wraplength=300)
        self.lbl_prediccion.pack(anchor="w", padx=8, pady=4)

        ctk.CTkLabel(left_stats, text="⚠️ Anomalías detectadas",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(14, 4))
        self.lbl_anomalias = ctk.CTkLabel(left_stats, text="Ninguna por ahora", justify="left", wraplength=300)
        self.lbl_anomalias.pack(anchor="w", padx=8)

        right_charts = ctk.CTkFrame(content)
        right_charts.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        self.chart_frame = ctk.CTkFrame(right_charts)
        self.chart_frame.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas_chart = None

    def _construir_tab_historial(self):
        frame = ctk.CTkFrame(self.tab_historial)
        frame.pack(fill="both", expand=True, padx=8, pady=8)

        top = ctk.CTkFrame(frame, fg_color="transparent")
        top.pack(fill="x", pady=(0, 6))

        ctk.CTkLabel(top, text="Historial de eventos y fotografías",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")

        self.btn_fotos = ctk.CTkButton(top, text="📷 Abrir galería de fotos", command=self.abrir_galeria, width=200)
        self.btn_fotos.pack(side="right", padx=6)

        self.historial_texto = ctk.CTkTextbox(frame, font=ctk.CTkFont(size=12))
        self.historial_texto.pack(fill="both", expand=True, padx=4, pady=4)
        self.historial_texto.insert("1.0", "Presioná 'Actualizar estadísticas' o iniciá la cámara para ver eventos.\n")

    # ==================================================================
    # ESTADÍSTICAS Y PREDICCIÓN
    # ==================================================================
    def cargar_eventos(self, dias=7):
        eventos = []
        hoy = datetime.now().date()

        for i in range(dias):
            fecha = hoy - timedelta(days=i)
            carpeta = os.path.join(self.carpeta_registros, fecha.strftime("%Y-%m-%d"))
            archivo = os.path.join(carpeta, "eventos.jsonl")
            if os.path.exists(archivo):
                try:
                    with open(archivo, "r", encoding="utf-8") as f:
                        for linea in f:
                            linea = linea.strip()
                            if linea:
                                try:
                                    ev = json.loads(linea)
                                    ev["_fecha"] = fecha
                                    eventos.append(ev)
                                except json.JSONDecodeError:
                                    continue
                except Exception:
                    pass
        return eventos

    def actualizar_estadisticas(self):
        eventos = self.cargar_eventos(dias=7)

        if not eventos:
            self.lbl_total_personas.configure(text="👤 Personas: 0")
            self.lbl_total_vehiculos.configure(text="🚗 Vehículos: 0")
            self.lbl_total_animales.configure(text="🐕 Animales: 0")
            self.lbl_total_eventos.configure(text="📋 Total eventos: 0")
            self.lbl_genero.configure(text="Sin datos")
            self.lbl_edad.configure(text="Sin datos")
            self.lbl_prediccion.configure(text="No hay suficientes datos históricos")
            self.lbl_anomalias.configure(text="Ninguna")
            return

        total_personas = 0
        total_vehiculos = 0
        total_animales = 0
        objetos_counter = Counter()
        por_hora = defaultdict(int)
        por_hora_personas = defaultdict(int)

        vehiculos_keys = {"auto", "moto", "camión", "colectivo", "bicicleta", "tren", "car", "motorcycle", "truck", "bus", "bicycle"}
        animales_keys = {"perro", "gato", "pájaro", "dog", "cat", "bird"}

        for ev in eventos:
            total_personas += ev.get("personas_detectadas", 0) or 0

            for obj in ev.get("objetos", []):
                nombre = str(obj.get("nombre", "")).lower()
                cant = obj.get("cantidad", 1)
                objetos_counter[nombre] += cant
                if nombre in vehiculos_keys:
                    total_vehiculos += cant
                elif nombre in animales_keys:
                    total_animales += cant

            try:
                dt = datetime.fromisoformat(ev.get("fecha_hora", ""))
                hora = dt.hour
                por_hora[hora] += 1
                if ev.get("personas_detectadas", 0) > 0:
                    por_hora_personas[hora] += ev.get("personas_detectadas", 0)
            except Exception:
                pass

        self.lbl_total_personas.configure(text=f"👤 Personas: {total_personas}")
        self.lbl_total_vehiculos.configure(text=f"🚗 Vehículos: {total_vehiculos}")
        self.lbl_total_animales.configure(text=f"🐕 Animales: {total_animales}")
        self.lbl_total_eventos.configure(text=f"📋 Total eventos: {len(eventos)}")

        self.lbl_genero.configure(text="Datos de género se irán acumulando\ncon más detecciones de rostros")
        self.lbl_edad.configure(text="Se calculará cuando se guarden más análisis faciales")

        hora_actual = datetime.now().hour
        prediccion_texto = self._generar_prediccion(por_hora_personas, por_hora, hora_actual)
        self.lbl_prediccion.configure(text=prediccion_texto)

        anomalias = self._detectar_anomalias(por_hora, hora_actual)
        self.lbl_anomalias.configure(text=anomalias if anomalias else "Ninguna anomalía detectada")

        self._dibujar_graficos(por_hora, objetos_counter)
        self._actualizar_texto_historial(eventos)

        self.status.configure(text=f"Estadísticas actualizadas – {len(eventos)} eventos analizados")

    def _generar_prediccion(self, por_hora_personas, por_hora, hora_actual):
        if not por_hora:
            return "Datos insuficientes para predecir"

        horas_futuras = [(hora_actual + i) % 24 for i in range(1, 4)]
        lineas = ["Según el patrón de los últimos 7 días:\n"]

        for h in horas_futuras:
            promedio = por_hora_personas.get(h, 0) / max(1, 7)
            eventos_prom = por_hora.get(h, 0) / max(1, 7)
            if promedio > 0.3 or eventos_prom > 0.5:
                lineas.append(f"• {h:02d}:00 → ~{promedio:.1f} personas / {eventos_prom:.1f} eventos")
            else:
                lineas.append(f"• {h:02d}:00 → actividad baja")

        return "\n".join(lineas)

    def _detectar_anomalias(self, por_hora, hora_actual):
        if not por_hora:
            return ""

        valores = list(por_hora.values())
        if len(valores) < 3:
            return ""

        media = np.mean(valores)
        std = np.std(valores) if len(valores) > 1 else 0

        actual = por_hora.get(hora_actual, 0)
        if std > 0 and actual > media + 2 * std:
            return f"⚠️ Pico inusual a las {hora_actual:02d}:00\n({actual} eventos vs media {media:.1f})"
        return ""

    def _dibujar_graficos(self, por_hora, objetos_counter):
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=100)
        fig.patch.set_facecolor("#2b2b2b")

        ax1 = axes[0]
        horas = list(range(24))
        valores = [por_hora.get(h, 0) for h in horas]
        ax1.bar(horas, valores, color="#3498db", alpha=0.85)
        ax1.set_title("Actividad por hora del día", color="white", fontsize=11)
        ax1.set_xlabel("Hora", color="white")
        ax1.set_ylabel("Eventos", color="white")
        ax1.tick_params(colors="white")
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_facecolor("#1e1e1e")

        ax2 = axes[1]
        if objetos_counter:
            top = objetos_counter.most_common(8)
            nombres = [n.title() for n, _ in top]
            cantidades = [c for _, c in top]
            ax2.barh(nombres[::-1], cantidades[::-1], color="#2ecc71", alpha=0.85)
            ax2.set_title("Objetos más detectados", color="white", fontsize=11)
            ax2.tick_params(colors="white")
            ax2.set_facecolor("#1e1e1e")
        else:
            ax2.text(0.5, 0.5, "Sin objetos registrados", ha="center", va="center", color="white")
            ax2.set_facecolor("#1e1e1e")

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_chart = canvas
        plt.close(fig)

    def _actualizar_texto_historial(self, eventos):
        self.historial_texto.delete("1.0", "end")
        if not eventos:
            self.historial_texto.insert("1.0", "No hay eventos registrados todavía.\n")
            return

        eventos_ordenados = sorted(eventos, key=lambda e: e.get("fecha_hora", ""), reverse=True)[:80]
        for ev in eventos_ordenados:
            fecha = ev.get("fecha_hora", "")[:19].replace("T", " ")
            personas = ev.get("personas_detectadas", 0)
            objs = ", ".join(f"{o['cantidad']} {o['nombre']}" for o in ev.get("objetos", []))
            linea = f"[{fecha}]  Personas: {personas}"
            if objs:
                linea += f"  |  {objs}"
            linea += "\n"
            self.historial_texto.insert("end", linea)

    # ==================================================================
    # MÉTODOS DE AUDIO RTSP
    # ==================================================================
    def iniciar_audio_rtsp(self, url):
        self.detener_audio_rtsp()
        try:
            self.audio_process = subprocess.Popen(
                [
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-rtsp_transport", "tcp",
                    "-probesize", "32",
                    "-analyzeduration", "0",
                    "-loglevel", "quiet",
                    url
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"No se pudo iniciar audio RTSP: {e}")

    def detener_audio_rtsp(self):
        if self.audio_process:
            try:
                self.audio_process.terminate()
                self.audio_process.wait(timeout=1)
            except Exception:
                try:
                    self.audio_process.kill()
                except Exception:
                    pass
            self.audio_process = None

    def grabar_audio_persona(self, url, duracion=8):
        ahora = time.time()
        if ahora - self.ultimo_audio_guardado < self.intervalo_audio or self.grabando_audio:
            return

        self.grabando_audio = True
        self.ultimo_audio_guardado = ahora

        try:
            ahora_dt = datetime.now()
            carpeta = os.path.join(self.carpeta_registros, ahora_dt.strftime("%Y-%m-%d"))
            os.makedirs(carpeta, exist_ok=True)

            nombre = f"audio_{ahora_dt.strftime('%H-%M-%S')}.mp3"
            ruta_audio = os.path.join(carpeta, nombre)

            comando = [
                "ffmpeg",
                "-y",
                "-rtsp_transport", "tcp",
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-i", url,
                "-t", str(duracion),
                "-vn",
                "-acodec", "libmp3lame",
                "-q:a", "4",
                "-loglevel", "error",
                ruta_audio
            ]

            def _grabar():
                try:
                    subprocess.run(comando, timeout=duracion + 5)
                    self.after(0, lambda: self.status.configure(text=f"Audio guardado: {nombre}"))
                except Exception as e:
                    print(f"Error al grabar audio: {e}")
                finally:
                    self.grabando_audio = False

            threading.Thread(target=_grabar, daemon=True).start()
        except Exception as e:
            print(f"Error preparando grabación de audio: {e}")
            self.grabando_audio = False

    # ==================================================================
    # CÁMARA Y PROCESAMIENTO (CON HILO DE CAPTURA INDEPENDIENTE)
    # ==================================================================
    def cambiar_tipo_camara(self, choice):
        if "RTSP" in choice or "Seguridad" in choice:
            self.label_rtsp.pack(pady=(6, 2))
            self.entry_rtsp.pack(pady=2)
        else:
            self.label_rtsp.pack_forget()
            self.entry_rtsp.pack_forget()

    def zoom_in(self):
        self.zoom_factor = min(3.0, round(self.zoom_factor + 0.2, 1))
        self.actualizar_controles_zoom()

    def zoom_out(self):
        self.zoom_factor = max(1.0, round(self.zoom_factor - 0.2, 1))
        self.actualizar_controles_zoom()

    def cambiar_zoom_slider(self, valor):
        self.zoom_factor = round(float(valor), 1)
        self.actualizar_controles_zoom()

    def actualizar_controles_zoom(self):
        self.zoom_label.configure(text=f"{self.zoom_factor:.1f}x")
        self.zoom_slider.set(self.zoom_factor)

    def aplicar_zoom(self, frame):
        if self.zoom_factor <= 1.0 or frame is None:
            return frame
        try:
            h, w = frame.shape[:2]
            nh, nw = max(1, int(h / self.zoom_factor)), max(1, int(w / self.zoom_factor))
            y1 = (h - nh) // 2
            x1 = (w - nw) // 2
            recorte = frame[y1:y1+nh, x1:x1+nw]
            return cv2.resize(recorte, (w, h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return frame

    def registrar_evento(self, faces, objects, frame=None, alturas_personas_px=None):
        try:
            alturas_personas_px = alturas_personas_px or []
            faces = faces or []
            objects = [o for o in (objects or []) if str(o).lower() not in ("person", "persona")]

            if not objects and not faces and not alturas_personas_px:
                return

            ahora_epoch = time.time()
            contador = Counter(objects)
            hay_persona = bool(faces or alturas_personas_px)

            if not hay_persona:
                if ahora_epoch - self.ultimo_guardado < 25.0 and Counter(self.ultimos_objetos) == contador:
                    return
                if ahora_epoch - self.ultimo_guardado < self.intervalo_registro:
                    return
            else:
                if ahora_epoch - self.ultimo_registro_persona < self.intervalo_registro_persona:
                    return

            self.ultimos_objetos = list(objects)
            ahora = datetime.now()
            hora = ahora.strftime("%d/%m/%Y %H:%M:%S")

            texto = f"[{hora}] 🔔 Evento:\n"
            if objects:
                texto += "   🚗/📦 " + ", ".join(f"{c} {self.traducir_objeto(n)}" for n, c in contador.items()) + "\n"

            total_personas = max(len(faces), len(alturas_personas_px))
            if total_personas > 0:
                texto += f"   👤 Personas: {total_personas}\n"
                if faces:
                    for i, cara in enumerate(faces, 1):
                        try:
                            edad = int(round(float(cara.get("age", 0))))
                        except Exception:
                            edad = "?"
                        genero = self.traducir_genero(cara.get("dominant_gender", "?"))
                        emocion = self.traducir_emocion(cara.get("dominant_emotion", "?"))
                        texto += f"      • P{i}: {edad}a | {genero} | {emocion}\n"

                tipo = self.tipo_camara.get()
                if "RTSP" in tipo or "Seguridad" in tipo:
                    url = self.entry_rtsp.get().strip()
                    if url:
                        self.grabar_audio_persona(url, duracion=8)
                            
            carpeta = os.path.join(self.carpeta_registros, ahora.strftime("%Y-%m-%d"))
            os.makedirs(carpeta, exist_ok=True)
            base = "evento_" + ahora.strftime("%H-%M-%S-%f")[:12]
            foto = os.path.join(carpeta, base + ".jpg")

            if frame is not None:
                try:
                    # Dibujar la fecha y hora exacta en la esquina inferior de la foto guardada
                    cv2.putText(frame, hora, (20, frame.shape[0] - 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    cv2.imwrite(foto, frame)
                    texto += "   📷 Foto guardada con recuadros y hora\n"
                except Exception as e:
                    print(f"Error al guardar foto: {e}") 

            datos = {
                "fecha_hora": ahora.isoformat(timespec="seconds"),
                "fotografia": foto if os.path.exists(foto) else None,
                "objetos": [{"nombre": self.traducir_objeto(k), "cantidad": v} for k, v in contador.items()],
                "personas_detectadas": total_personas,
                "alturas_cm": [self.estimar_altura(px) for px in alturas_personas_px],
            }
            if faces:
                datos["generos"] = [self.traducir_genero(c.get("dominant_gender", "?")) for c in faces]
                datos["edades"] = []
                for c in faces:
                    try:
                        datos["edades"].append(int(round(float(c.get("age", 0)))))
                    except Exception:
                        pass

            try:
                with open(os.path.join(carpeta, "eventos.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(datos, ensure_ascii=False) + "\n")
            except Exception:
                pass

            self.ultimo_guardado = ahora_epoch
            if hay_persona:
                self.ultimo_registro_persona = ahora_epoch

            texto += "-" * 40 + "\n"
            self.after(0, lambda: self._insertar_historial_forzado(texto, hora))

        except Exception as e:
            print(f"Error en registrar_evento: {e}")

    def _insertar_historial_forzado(self, texto, hora):
        try:
            self.resultado.insert("1.0", texto)
            self.status.configure(text=f"Último evento: {hora}")
            self.reproducir_sonido_camara()
        except Exception:
            pass

    def limpiar_registro(self):
        self.resultado.delete("1.0", "end")
        self.resultado.insert("1.0", "Registro limpiado.\n")
        self.ultimos_objetos = []

    def cargar_img1(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.img1_path = path
            self.btn_img1.configure(text=f"✅ {os.path.basename(path)[:18]}")

    def cargar_img2(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.img2_path = path
            self.btn_img2.configure(text=f"✅ {os.path.basename(path)[:18]}")

    def iniciar_verificacion(self):
        if not self.img1_path or not self.img2_path:
            messagebox.showwarning("Faltan imágenes", "Carga las dos imágenes.")
            return
        threading.Thread(target=self.verificar, daemon=True).start()

    def verificar(self):
        try:
            res = DeepFace.verify(self.img1_path, self.img2_path, model_name=self.modelo.get(), enforce_detection=False)
            texto = f"[{datetime.now().strftime('%H:%M:%S')}] VERIFICACIÓN: {'✅ Misma persona' if res['verified'] else '❌ Diferentes'}\n"
            self.after(0, lambda: self.resultado.insert("1.0", texto + "-" * 40 + "\n"))
        except Exception as e:
            self.after(0, lambda: self.resultado.insert("1.0", f"Error: {e}\n"))

    def traducir_objeto(self, nombre):
        mapa = {
            "person": "persona", "bicycle": "bicicleta", "car": "auto", "motorcycle": "moto",
            "bus": "colectivo", "truck": "camión", "train": "tren", "traffic light": "semáforo",
            "stop sign": "señal de pare", "fire hydrant": "hidrante", "parking meter": "parquímetro",
            "bench": "banco", "dog": "perro", "cat": "gato", "bird": "pájaro", "backpack": "mochila",
            "handbag": "cartera", "umbrella": "paraguas", "suitcase": "valija", "bottle": "botella",
            "cell phone": "celular", "laptop": "notebook", "book": "libro", "cup": "vaso",
            "sports ball": "pelota", "skateboard": "skate", "mate": "mate", "termo": "termo",
            "headphones": "auriculares", "sunglasses": "lentes de sol", "wallet": "billetera",
            "keys": "llaves", "cigarette": "cigarrillo", "cap": "gorra", "watch": "reloj",
            "potted plant": "planta en maceta", "chair": "silla", "couch": "sillón", "bed": "cama",
            "dining table": "mesa", "toilet": "inodoro", "tv": "televisor", "mouse": "mouse",
            "remote": "control remoto", "keyboard": "teclado", "microwave": "microondas",
            "oven": "horno", "toaster": "tostadora", "sink": "bachera", "refrigerator": "heladera",
            "clock": "reloj", "vase": "florero", "scissors": "tijeras", "teddy bear": "peluche",
            "hair drier": "secador de pelo", "toothbrush": "cepillo de dientes", "wine glass": "copa",
            "fork": "tenedor", "knife": "cuchillo", "spoon": "cuchara", "bowl": "bowl",
            "banana": "banana", "apple": "manzana", "sandwich": "sándwich", "orange": "naranja",
            "broccoli": "brócoli", "carrot": "zanahoria", "hot dog": "pancho", "pizza": "pizza",
            "donut": "dona", "cake": "torta"
        }
        return mapa.get(str(nombre).strip().lower().replace("_", " "), str(nombre).lower())

    def traducir_genero(self, valor):
        if isinstance(valor, dict):
            valor = max(valor, key=valor.get) if valor else ""
        return {"man": "masculino", "woman": "femenino", "male": "masculino", "female": "femenino"}.get(
            str(valor).strip().lower(), "no disponible")

    def traducir_emocion(self, valor):
        if isinstance(valor, dict):
            valor = max(valor, key=valor.get) if valor else ""
        return {"angry": "enojo", "disgust": "disgusto", "fear": "miedo", "happy": "alegría",
                "sad": "tristeza", "surprise": "sorpresa", "neutral": "neutral"}.get(
            str(valor).strip().lower(), str(valor))

    def obtener_factor_altura(self):
        try:
            v = self.entry_factor_altura.get().strip().replace(",", ".")
            return float(v) if v else 0.45
        except Exception:
            return 0.45

    def estimar_altura(self, px):
        try:
            return round(float(px) * self.obtener_factor_altura())
        except Exception:
            return None

    def reproducir_sonido_camara(self):
        try:
            if self.sonido_activo.get():
                if os.name == "nt":
                    import winsound
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
                else:
                    self.bell()
        except Exception:
            pass

    def abrir_galeria(self):
        ventana = ctk.CTkToplevel(self)
        ventana.title("Galería de fotografías")
        ventana.geometry("900x650")
        marco = ctk.CTkScrollableFrame(ventana)
        marco.pack(fill="both", expand=True, padx=10, pady=10)

        fotos = []
        for raiz, _, archivos in os.walk(self.carpeta_registros):
            for n in archivos:
                if n.lower().endswith((".jpg", ".jpeg", ".png")):
                    fotos.append(os.path.join(raiz, n))
        fotos.sort(key=os.path.getmtime, reverse=True)

        if not fotos:
            ctk.CTkLabel(marco, text="Todavía no hay fotografías.").pack(pady=30)
            return

        for ruta in fotos[:60]:
            fila = ctk.CTkFrame(marco)
            fila.pack(fill="x", padx=4, pady=4)
            try:
                img = Image.open(ruta).convert("RGB")
                img.thumbnail((140, 90))
                foto = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                lbl = ctk.CTkLabel(fila, image=foto, text="")
                lbl.image = foto
                lbl.pack(side="left", padx=6)
            except Exception:
                pass
            ctk.CTkLabel(fila, text=os.path.basename(ruta)).pack(side="left", padx=6)
            ctk.CTkButton(fila, text="Ver", width=70,
                          command=lambda r=ruta: self.ver_fotografia(r)).pack(side="right", padx=8)

    def ver_fotografia(self, ruta):
        try:
            img = Image.open(ruta).convert("RGB")
            ventana = ctk.CTkToplevel(self)
            ventana.title(os.path.basename(ruta))
            ventana.geometry("920x700")
            img.thumbnail((880, 620))
            foto = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            lbl = ctk.CTkLabel(ventana, image=foto, text="")
            lbl.image = foto
            lbl.pack(expand=True, padx=10, pady=10)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def toggle_camara(self):
        if self.camera_running:
            self.detener_camara()
        else:
            self.iniciar_camara()

    def iniciar_camara(self):
        tipo = self.tipo_camara.get()
        if "RTSP" in tipo or "Seguridad" in tipo:
            url = self.entry_rtsp.get().strip()
            if not url:
                messagebox.showwarning("URL vacía", "Escribe la URL RTSP.")
                return
            fuente = url
            self.iniciar_audio_rtsp(url)
        else:
            fuente = 0

        if isinstance(fuente, str):
            self.cap = cv2.VideoCapture(fuente, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(fuente, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(fuente)

        self.es_webcam = not isinstance(fuente, str)
        if isinstance(fuente, str) and "rtsp" in str(fuente).lower():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo conectar a la cámara.")
            return

        self.camera_running = True
        self.frame_count = 0
        self.persistent_boxes.clear()
        self.btn_camara.configure(text="⏹️ Detener Cámara", fg_color="#c0392b")
        self.status.configure(text="Cámara activa")

        # Hilos independientes para lectura fluida y análisis separado
        threading.Thread(target=self._hilo_captura_rtsp, daemon=True).start()
        threading.Thread(target=self.actualizar_camara, daemon=True).start()
        threading.Thread(target=self._hilo_analisis_deepface_seguro, daemon=True).start()

    def detener_camara(self):
        self.camera_running = False
        self.detener_audio_rtsp()

        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_camara.configure(text="▶️ Iniciar Cámara", fg_color="#e67e22")
        self.camera_label.configure(image=None, text="Cámara apagada")
        self.status.configure(text="Cámara detenida")

    def _hilo_captura_rtsp(self):
        """Hilo dedicado exclusivamente a capturar el frame más fresco de la red sin bloquear la UI ni YOLO"""
        while self.camera_running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

                with self.frame_lock:
                    self.latest_raw_frame = frame
            except Exception:
                time.sleep(0.01)

    def _hilo_analisis_deepface_seguro(self):
        while self.camera_running:
            try:
                with self.analysis_lock:
                    if self.current_frame_for_analysis is None:
                        time.sleep(0.3)
                        continue
                    frame_copia = self.current_frame_for_analysis.copy()
                    alturas = list(self.current_alturas)

                if not alturas:
                    with self.analysis_lock:
                        self.current_faces = []
                    time.sleep(0.8)
                    continue

                rgb = cv2.cvtColor(frame_copia, cv2.COLOR_BGR2RGB)
                analisis = DeepFace.analyze(rgb, actions=['age', 'gender', 'emotion'],
                                            detector_backend='opencv', enforce_detection=False, silent=True)
                if isinstance(analisis, dict):
                    analisis = [analisis]
                validos = []
                for c in analisis:
                    if c.get('age') is not None:
                        reg = c.get('region') or c.get('facial_area', {})
                        if reg and reg.get('w', 0) >= 30 and reg.get('h', 0) >= 30:
                            c['facial_area'] = reg
                            validos.append(c)

                with self.analysis_lock:
                    self.current_faces = validos
            except Exception:
                pass
            time.sleep(1.1)
            
    def actualizar_camara(self):
        """Bucle principal de procesamiento visual optimizado con el frame más reciente"""
        while self.camera_running:
            try:
                with self.frame_lock:
                    if self.latest_raw_frame is None:
                        time.sleep(0.01)
                        continue
                    frame = self.latest_raw_frame.copy()

                self.frame_count += 1
                if self.es_webcam:
                    frame = cv2.flip(frame, 1)
                frame = self.aplicar_zoom(frame)

                with self.analysis_lock:
                    self.current_frame_for_analysis = frame.copy()

                objetos_detectados = []
                alturas_personas_px = []
                boxes_to_draw = []

                if self.yolo_model is not None:
                    try:
                        # Ejecutar YOLO cada 4 frames optimizado
                        if self.frame_count % 4 == 0:
                            results = self.yolo_model.track(
                                frame, persist=True, verbose=False, conf=0.45, imgsz=320
                            )
                        else:
                            results = []

                        nuevas_posiciones_objetos = {}
                        
                        if not hasattr(self, "memoria_objetos_estaticos"):
                            self.memoria_objetos_estaticos = {}

                        for r in results:
                            if r.boxes is None:
                                continue
                            for box in r.boxes:
                                cls_id = int(box.cls[0])
                                nombre = str(self.yolo_model.names[cls_id]).lower()
                                conf = float(box.conf[0])
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                h_box = max(1, y2 - y1)
                                w_box = max(1, x2 - x1)
                                aspect = h_box / float(w_box)

                                if nombre == "person":
                                    if (conf >= self.person_conf_min and
                                        aspect >= self.person_aspect_min and
                                        h_box >= self.person_min_height_px):
                                        alturas_personas_px.append(h_box)
                                        boxes_to_draw.append(("person", x1, y1, x2, y2, "Persona"))
                                else:
                                    if conf >= 0.40:
                                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                        match_encontrado = None
                                        
                                        for obj_id, datos in self.memoria_objetos_estaticos.items():
                                            if datos["nombre"] == nombre:
                                                distancia = np.hypot(cx - datos["cx"], cy - datos["cy"])
                                                if distancia < 30: 
                                                    match_encontrado = obj_id
                                                    break

                                        if match_encontrado is not None:
                                            frames_staticos = self.memoria_objetos_estaticos[match_encontrado]["frames_quieto"] + 1
                                            nuevas_posiciones_objetos[match_encontrado] = {
                                                "nombre": nombre, "cx": cx, "cy": cy, "frames_quieto": frames_staticos
                                            }
                                        else:
                                            nuevo_id = len(nuevas_posiciones_objetos) + 1000
                                            nuevas_posiciones_objetos[nuevo_id] = {
                                                "nombre": nombre, "cx": cx, "cy": cy, "frames_quieto": 0
                                            }
                                            frames_staticos = 0

                                        if frames_staticos > self.static_threshold:
                                            continue  

                                        objetos_detectados.append(nombre)
                                        boxes_to_draw.append((nombre, x1, y1, x2, y2, self.traducir_objeto(nombre)))

                        self.memoria_objetos_estaticos = nuevas_posiciones_objetos

                    except Exception as e:
                        print(f"Error YOLO Tracking: {e}")

                with self.analysis_lock:
                    self.current_alturas = list(alturas_personas_px)
                    self.current_objects = list(objetos_detectados)
                    faces_actuales = list(self.current_faces)

                display = frame.copy()
                for item in boxes_to_draw:
                    nombre, x1, y1, x2, y2, etiqueta = item
                    color = (0, 220, 100) if nombre == "person" else (255, 160, 0)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display, etiqueta, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                for i, px in enumerate(alturas_personas_px):
                    cm = self.estimar_altura(px)
                    if cm:
                        cv2.putText(display, f"~{cm} cm", (10, 30 + i * 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                texto_vivo = f"Frame: {self.frame_count}  |  Personas: {len(alturas_personas_px)}"
                if objetos_detectados:
                    cont = Counter(objetos_detectados)
                    texto_vivo += "  |  " + ", ".join(f"{c} {self.traducir_objeto(n)}" for n, c in cont.items())
                if faces_actuales:
                    texto_vivo += f"\nCaras analizadas: {len(faces_actuales)}"
                    for i, cara in enumerate(faces_actuales[:3], 1):
                        try:
                            edad = int(round(float(cara.get("age", 0))))
                        except Exception:
                            edad = "?"
                        genero = self.traducir_genero(cara.get("dominant_gender", "?"))
                        emocion = self.traducir_emocion(cara.get("dominant_emotion", "?"))
                        texto_vivo += f"\n  P{i}: {edad}a | {genero} | {emocion}"

                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                self.after(0, lambda r=rgb, t=texto_vivo: self._actualizar_vista_ui(r, t))

                if alturas_personas_px or objetos_detectados or faces_actuales:
                    self.registrar_evento(
                        faces=faces_actuales,
                        objects=objetos_detectados,
                        frame=frame,
                        alturas_personas_px=alturas_personas_px
                    )

            except Exception as e:
                print(f"Error en actualizar_camara: {e}")
                time.sleep(0.01)
                continue

            time.sleep(0.01)

    def _actualizar_vista_ui(self, rgb, texto):
        if not self.winfo_exists():
            return
        try:
            img = Image.fromarray(rgb).resize((640, 400), Image.BILINEAR)
            photo = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 400))
            self.camera_label.configure(image=photo, text="")
            self.camera_label.image = photo
            self.live_result.delete("1.0", "end")
            self.live_result.insert("1.0", texto)
        except Exception:
            pass

    def on_closing(self):
        self.camera_running = False
        self.detener_audio_rtsp()
        if self.cap:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = DeepFaceApp()
    app.mainloop()