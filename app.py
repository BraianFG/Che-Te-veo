# app.py - DeepFace + YOLO con PTZ y Historial Rediseñado
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

try:
    from onvif import ONVIFCamera
except ImportError:
    ONVIFCamera = None

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

plt.style.use("dark_background")


class DeepFaceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Che, Te Veo - Sistema de Monitoreo Inteligente")
        
     # Definimos el tamaño base y aseguramos que inicie maximizada
        self.geometry("1520x940")
        self.minsize(1350, 850)
        
        # Forzar pantalla maximizada de forma nativa en Windows sin conflictos
        self.after(100, lambda: self.state("zoomed"))

        self.img1_path = None
        self.img2_path = None

        self.audio_process = None
        self.grabando_audio = False
        self.ultimo_audio_guardado = 0.0
        self.intervalo_audio = 10.0
        
        self.cap = None
        self.camera_running = False
        self.frame_count = 0
        self.current_faces = []
        self.current_objects = []
        self.current_alturas = []
        
        self.onvif_client = None
        self.ptz_service = None
        self.media_service = None
        self.token_profile = None
        self.ultimo_movimiento_ptz = 0.0

        self.latest_raw_frame = None
        self.frame_lock = threading.Lock()

        self.ultimos_objetos = []
        self.ultimo_guardado = 0.0
        self.intervalo_registro = 2.0
        self.intervalo_registro_persona = 8.0
        self.ultimo_registro_persona = 0.0
        self.es_webcam = True
        self.zoom_factor = 1.0

        self.person_conf_min = 0.50
        self.person_aspect_min = 0.50
        self.person_min_height_px = 0.50

        self.persistent_boxes = deque(maxlen=2)
        self.current_frame_for_analysis = None
        self.analysis_lock = threading.Lock()

        self.carpeta_registros = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registros")
        os.makedirs(self.carpeta_registros, exist_ok=True)

        try:
            self.yolo_model = YOLO("yolov8n.pt")
        except Exception:
            try:
                self.yolo_model = YOLO("yolov8s.pt")
            except Exception as e2:
                messagebox.showerror("Error", f"No se pudo cargar YOLO:\n{e2}")
                self.yolo_model = None

        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=15, pady=(10, 0))

        self.label_titulo = ctk.CTkLabel(
            header_frame, text="🎭 Che, Te Veo",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.label_titulo.pack(side="left")

        self.tabview = ctk.CTkTabview(self, width=1480, height=840)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=5)

        self.tab_vivo = self.tabview.add("📹 En Vivo")
        self.tab_stats = self.tabview.add("📊 Estadísticas")
        self.tab_historial = self.tabview.add("📋 Historial")

        self._construir_tab_vivo()
        self._construir_tab_estadisticas()
        self._construir_tab_historial()

        self.status = ctk.CTkLabel(self, text="Sistema Listo", font=ctk.CTkFont(size=12), text_color="#aaaaaa")
        self.status.pack(pady=(2, 6))

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.cambiar_tipo_camara("Cámara Local (Webcam)")

    def _construir_tab_vivo(self):
        main_frame = ctk.CTkFrame(self.tab_vivo, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=4, pady=4)

        left = ctk.CTkScrollableFrame(main_frame, width=340, corner_radius=10)
        left.pack(side="left", fill="y", padx=(0, 6), pady=4)

        card_cam = ctk.CTkFrame(left, fg_color="#1e1e1e", corner_radius=8)
        card_cam.pack(fill="x", pady=4, padx=4)

        ctk.CTkLabel(card_cam, text="📹 Fuente de Cámara", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=10, pady=(8, 4))
        self.tipo_camara = ctk.CTkComboBox(
            card_cam, values=["Cámara Local (Webcam)", "Cámara de Seguridad (RTSP)"],
            width=300, command=self.cambiar_tipo_camara
        )
        self.tipo_camara.set("Cámara Local (Webcam)")
        self.tipo_camara.pack(pady=4, padx=10)

        self.label_rtsp = ctk.CTkLabel(card_cam, text="URL RTSP:", font=ctk.CTkFont(size=11))
        self.entry_rtsp = ctk.CTkEntry(card_cam, width=300, placeholder_text="rtsp://admin:clave@192.168.1.105:554/onvif1")

        self.btn_camara = ctk.CTkButton(
            card_cam, text="▶️ Iniciar Cámara", command=self.toggle_camara,
            fg_color="#e67e22", hover_color="#d35400", width=300, height=36,
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.btn_camara.pack(pady=(10, 10), padx=10)

        self.card_ptz = ctk.CTkFrame(left, fg_color="#1e1e1e", corner_radius=8)
        ctk.CTkLabel(self.card_ptz, text="🕹️ Control PTZ 360°", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=10, pady=(8, 4))
        
        frame_ip_port = ctk.CTkFrame(self.card_ptz, fg_color="transparent")
        frame_ip_port.pack(fill="x", padx=10)
        self.entry_onvif_ip = ctk.CTkEntry(frame_ip_port, width=190, placeholder_text="IP (192.168.1.105)")
        self.entry_onvif_ip.pack(side="left", padx=(0, 4))
        self.entry_onvif_port = ctk.CTkEntry(frame_ip_port, width=100, placeholder_text="Puerto (80)")
        self.entry_onvif_port.pack(side="left")

        frame_user_pass = ctk.CTkFrame(self.card_ptz, fg_color="transparent")
        frame_user_pass.pack(fill="x", padx=10, pady=4)
        self.entry_onvif_user = ctk.CTkEntry(frame_user_pass, width=145, placeholder_text="Usuario")
        self.entry_onvif_user.pack(side="left", padx=(0, 4))
        self.entry_onvif_pass = ctk.CTkEntry(frame_user_pass, width=145, show="*", placeholder_text="Contraseña")
        self.entry_onvif_pass.pack(side="left")

        self.switch_ptz_auto = ctk.CTkSwitch(self.card_ptz, text="🔄 Tracking PTZ Automático")
        self.switch_ptz_auto.select()
        self.switch_ptz_auto.pack(pady=6, padx=10, anchor="w")

        ctk.CTkLabel(self.card_ptz, text="Control Manual:", font=ctk.CTkFont(size=11), text_color="#aaaaaa").pack(anchor="w", padx=10)
        dpad_frame = ctk.CTkFrame(self.card_ptz, fg_color="transparent")
        dpad_frame.pack(pady=(2, 10))

        ctk.CTkButton(dpad_frame, text="▲", width=38, height=32, command=lambda: self.mover_ptz_manual(0.0, 0.4)).grid(row=0, column=1, pady=2)
        ctk.CTkButton(dpad_frame, text="◀", width=38, height=32, command=lambda: self.mover_ptz_manual(-0.4, 0.0)).grid(row=1, column=0, padx=2)
        ctk.CTkButton(dpad_frame, text="▶", width=38, height=32, command=lambda: self.mover_ptz_manual(0.4, 0.0)).grid(row=1, column=2, padx=2)
        ctk.CTkButton(dpad_frame, text="▼", width=38, height=32, command=lambda: self.mover_ptz_manual(0.0, -0.4)).grid(row=2, column=1, pady=2)

        card_det = ctk.CTkFrame(left, fg_color="#1e1e1e", corner_radius=8)
        card_det.pack(fill="x", pady=4, padx=4)

        ctk.CTkLabel(card_det, text="⚙️ Configuración de Detección", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=10, pady=(8, 4))
        ctk.CTkLabel(card_det, text="Modelo Facial:", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=10)
        self.modelo = ctk.CTkComboBox(card_det, values=["VGG-Face", "Facenet", "ArcFace", "SFace"], width=300)
        self.modelo.set("VGG-Face")
        self.modelo.pack(pady=2, padx=10)

        ctk.CTkLabel(card_det, text="Calibración altura (cm/píxel):", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=10, pady=(4, 0))
        self.entry_factor_altura = ctk.CTkEntry(card_det, width=300, placeholder_text="Ej: 0.45")
        self.entry_factor_altura.pack(pady=2, padx=10)

        self.sonido_activo = ctk.BooleanVar(value=True)
        self.switch_sonido = ctk.CTkSwitch(card_det, text="🔔 Sonido al registrar", variable=self.sonido_activo)
        self.switch_sonido.pack(pady=(8, 8), anchor="w", padx=10)

        card_ver = ctk.CTkFrame(left, fg_color="#1e1e1e", corner_radius=8)
        card_ver.pack(fill="x", pady=4, padx=4)

        ctk.CTkLabel(card_ver, text="🔍 Verificación Facial", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=10, pady=(8, 4))
        self.btn_img1 = ctk.CTkButton(card_ver, text="📁 Cargar Imagen 1", command=self.cargar_img1, width=300, fg_color="#2980b9")
        self.btn_img1.pack(pady=2, padx=10)
        self.btn_img2 = ctk.CTkButton(card_ver, text="📁 Cargar Imagen 2", command=self.cargar_img2, width=300, fg_color="#2980b9")
        self.btn_img2.pack(pady=2, padx=10)
        self.btn_verificar = ctk.CTkButton(card_ver, text="🔍 Comparar Rostros", command=self.iniciar_verificacion,
                                           fg_color="#27ae60", hover_color="#1e8449", width=300)
        self.btn_verificar.pack(pady=(6, 10), padx=10)

        center = ctk.CTkFrame(main_frame, fg_color="#181818", corner_radius=10)
        center.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        header = ctk.CTkFrame(center, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=6)
        ctk.CTkLabel(header, text="Vista Principal", font=ctk.CTkFont(size=14, weight="bold")).pack(side="left")

        zoom_frame = ctk.CTkFrame(header, fg_color="transparent")
        zoom_frame.pack(side="right")
        ctk.CTkButton(zoom_frame, text="-", width=28, height=24, command=self.zoom_out).pack(side="left", padx=2)
        self.zoom_label = ctk.CTkLabel(zoom_frame, text="1.0x", width=40)
        self.zoom_label.pack(side="left")
        ctk.CTkButton(zoom_frame, text="+", width=28, height=24, command=self.zoom_in).pack(side="left", padx=2)

        self.camera_label = ctk.CTkLabel(center, text="Cámara Apagada", width=720, height=450,
                                         fg_color="#0d0d0d", corner_radius=8, font=ctk.CTkFont(size=16))
        self.camera_label.pack(pady=4, padx=10, fill="both", expand=True)

        self.zoom_slider = ctk.CTkSlider(center, from_=1.0, to=3.0, number_of_steps=20,
                                         command=self.cambiar_zoom_slider, width=650)
        self.zoom_slider.set(1.0)
        self.zoom_slider.pack(pady=4)

        self.live_result = ctk.CTkTextbox(center, height=80, font=ctk.CTkFont(size=12), fg_color="#121212")
        self.live_result.pack(pady=(2, 8), padx=10, fill="x")
        self.live_result.insert("1.0", "Esperando inicio de cámara...")

        right = ctk.CTkFrame(main_frame, width=300, fg_color="#181818", corner_radius=10)
        right.pack(side="right", fill="y", padx=(6, 0), pady=4)
        right.pack_propagate(False)

        ctk.CTkLabel(right, text="📋 Registro en Tiempo Real", font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(10, 4), padx=10, anchor="w")
        self.resultado = ctk.CTkTextbox(right, font=ctk.CTkFont(size=11), fg_color="#121212")
        self.resultado.pack(pady=4, padx=8, fill="both", expand=True)
        self.resultado.insert("1.0", "Eventos detectados aparecerán aquí...\n")

        self.btn_limpiar = ctk.CTkButton(right, text="🗑️ Limpiar Registro", command=self.limpiar_registro,
                                         fg_color="#555555", hover_color="#333333", width=260)
        self.btn_limpiar.pack(pady=8)

    def _construir_tab_estadisticas(self):
        frame = ctk.CTkFrame(self.tab_stats)
        frame.pack(fill="both", expand=True, padx=8, pady=8)

        top = ctk.CTkFrame(frame, fg_color="transparent")
        top.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(top, text="Análisis Estadístico del Sistema",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(side="left", padx=6)

        self.btn_actualizar_stats = ctk.CTkButton(
            top, text="🔄 Actualizar Datos", command=self.actualizar_estadisticas,
            fg_color="#2980b9", width=140, height=32
        )
        self.btn_actualizar_stats.pack(side="right", padx=6)

        self.label_periodo = ctk.CTkLabel(top, text="Últimos 7 días", text_color="#aaaaaa",
                                          font=ctk.CTkFont(size=13))
        self.label_periodo.pack(side="right", padx=12)

        kpi_row = ctk.CTkFrame(frame, fg_color="transparent")
        kpi_row.pack(fill="x", pady=(0, 10))

        def crear_kpi(parent, icono, titulo, valor_inicial, color_icono):
            card = ctk.CTkFrame(parent, fg_color="#1e1e1e", corner_radius=10, height=80)
            card.pack(side="left", fill="x", expand=True, padx=4)
            card.pack_propagate(False)
            inner = ctk.CTkFrame(card, fg_color="transparent")
            inner.pack(fill="both", expand=True, padx=12, pady=10)
            ctk.CTkLabel(inner, text=icono, font=ctk.CTkFont(size=22),
                         text_color=color_icono).pack(side="left", padx=(0, 10))
            txt_frame = ctk.CTkFrame(inner, fg_color="transparent")
            txt_frame.pack(side="left", fill="both", expand=True)
            ctk.CTkLabel(txt_frame, text=titulo, font=ctk.CTkFont(size=12),
                         text_color="#aaaaaa", anchor="w").pack(anchor="w")
            lbl_valor = ctk.CTkLabel(txt_frame, text=valor_inicial,
                                     font=ctk.CTkFont(size=22, weight="bold"), anchor="w")
            lbl_valor.pack(anchor="w")
            return lbl_valor

        self.lbl_total_personas = crear_kpi(kpi_row, "👤", "Personas", "—", "#3498db")
        self.lbl_total_vehiculos = crear_kpi(kpi_row, "🚗", "Vehículos", "—", "#2ecc71")
        self.lbl_total_animales = crear_kpi(kpi_row, "🐕", "Animales", "—", "#9b59b6")
        self.lbl_total_eventos = crear_kpi(kpi_row, "📋", "Total eventos", "—", "#e67e22")

        content = ctk.CTkFrame(frame, fg_color="transparent")
        content.pack(fill="both", expand=True)

        left = ctk.CTkFrame(content, fg_color="transparent", width=420)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self.chart_hora_frame = ctk.CTkFrame(left, fg_color="#1e1e1e", corner_radius=10)
        self.chart_hora_frame.pack(fill="both", expand=True, pady=(0, 8))

        pred_card = ctk.CTkFrame(left, fg_color="#1e1e1e", corner_radius=10)
        pred_card.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(pred_card, text="🔮 Predicción próximas 3 horas",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=12, pady=(10, 4))
        self.lbl_prediccion = ctk.CTkLabel(pred_card,
                                           text="Actualizá las estadísticas para ver la predicción",
                                           justify="left", wraplength=380, anchor="w")
        self.lbl_prediccion.pack(anchor="w", padx=12, pady=(0, 10))

        anom_card = ctk.CTkFrame(left, fg_color="#1e1e1e", corner_radius=10)
        anom_card.pack(fill="x")
        ctk.CTkLabel(anom_card, text="⚠️ Anomalías Detectadas",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=12, pady=(10, 4))
        self.lbl_anomalias = ctk.CTkLabel(anom_card, text="Ninguna detectada",
                                          justify="left", wraplength=380, anchor="w",
                                          text_color="#2ecc71")
        self.lbl_anomalias.pack(anchor="w", padx=12, pady=(0, 10))

        right = ctk.CTkFrame(content, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True, padx=(6, 0))

        self.chart_objetos_frame = ctk.CTkFrame(right, fg_color="#1e1e1e", corner_radius=10)
        self.chart_objetos_frame.pack(fill="both", expand=True, pady=(0, 8))

        bottom_right = ctk.CTkFrame(right, fg_color="transparent")
        bottom_right.pack(fill="x")

        genero_card = ctk.CTkFrame(bottom_right, fg_color="#1e1e1e", corner_radius=10)
        genero_card.pack(side="left", fill="both", expand=True, padx=(0, 4))
        ctk.CTkLabel(genero_card, text="👥 Género Detectado",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=12, pady=(10, 4))
        self.lbl_genero = ctk.CTkLabel(genero_card, text="Sin datos suficientes",
                                       justify="left", wraplength=180, anchor="w", text_color="#aaaaaa")
        self.lbl_genero.pack(anchor="w", padx=12, pady=(0, 10))

        edad_card = ctk.CTkFrame(bottom_right, fg_color="#1e1e1e", corner_radius=10)
        edad_card.pack(side="left", fill="both", expand=True, padx=(4, 0))
        ctk.CTkLabel(edad_card, text="🎂 Edad Promedio",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=12, pady=(10, 4))
        self.lbl_edad = ctk.CTkLabel(edad_card, text="Sin datos suficientes",
                                 justify="left", wraplength=180, anchor="w", text_color="#aaaaaa")
        self.lbl_edad.pack(anchor="w", padx=12, pady=(0, 10))

        self.canvas_chart = None
        self.canvas_objetos = None

    def _construir_tab_historial(self):
        # Rediseño avanzado de la pestaña Historial con tarjetas visuales y filtros
        frame = ctk.CTkFrame(self.tab_historial, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=8, pady=8)

        # Barra superior de acciones y filtros
        top_bar = ctk.CTkFrame(frame, fg_color="#1e1e1e", corner_radius=10, height=60)
        top_bar.pack(fill="x", pady=(0, 8))
        top_bar.pack_propagate(False)

        inner_top = ctk.CTkFrame(top_bar, fg_color="transparent")
        inner_top.pack(fill="both", expand=True, padx=12, pady=10)

        ctk.CTkLabel(inner_top, text="📋 Historial Inteligente de Eventos", font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")

        self.btn_refrescar_historial = ctk.CTkButton(inner_top, text="🔄 Recargar Historial", command=self.poblar_historial_tarjetas, width=150, fg_color="#2980b9")
        self.btn_refrescar_historial.pack(side="right", padx=6)

        self.btn_fotos = ctk.CTkButton(inner_top, text="📷 Galería de Fotos", command=self.abrir_galeria, width=150, fg_color="#27ae60")
        self.btn_fotos.pack(side="right", padx=6)

        # Contenedor desplazable con tarjetas modernas de eventos
        self.historial_scroll_frame = ctk.CTkScrollableFrame(frame, fg_color="#141414", corner_radius=10)
        self.historial_scroll_frame.pack(fill="both", expand=True, padx=2, pady=2)

        # Mensaje inicial por defecto
        self.lbl_vacio_historial = ctk.CTkLabel(self.historial_scroll_frame, text="No hay eventos cargados. Hacé clic en 'Recargar Historial' para buscar registros.", text_color="#888888", font=ctk.CTkFont(size=13))
        self.lbl_vacio_historial.pack(pady=40)

    def poblar_historial_tarjetas(self):
        # Limpiar elementos anteriores en el scrollable frame
        for widget in self.historial_scroll_frame.winfo_children():
            widget.destroy()

        eventos = self.cargar_eventos(dias=7)
        if not eventos:
            ctk.CTkLabel(self.historial_scroll_frame, text="No se encontraron registros en los últimos 7 días.", text_color="#888888", font=ctk.CTkFont(size=13)).pack(pady=40)
            return

        # Mostrar cada evento en formato de tarjeta estructurada
        for ev in eventos[:50]:
            card = ctk.CTkFrame(self.historial_scroll_frame, fg_color="#1e1e1e", corner_radius=8)
            card.pack(fill="x", padx=6, pady=6)

            inner_card = ctk.CTkFrame(card, fg_color="transparent")
            inner_card.pack(fill="both", expand=True, padx=12, pady=10)

            # Información de fecha/hora y personas
            fh = ev.get("fecha_hora", "Desconocido")
            personas = ev.get("personas_detectadas", 0)
            
            header_lbl = ctk.CTkLabel(inner_card, text=f"📅 Fecha: {fh}   |   👤 Personas: {personas}", font=ctk.CTkFont(size=12, weight="bold"), text_color="#3498db")
            header_lbl.pack(anchor="w", pady=(0, 4))

            # Objetos detectados
            objs = ev.get("objetos", [])
            objs_str = ", ".join([f"{o['cantidad']} {o['nombre']}" for o in objs]) if objs else "Ninguno detectado"
            obj_lbl = ctk.CTkLabel(inner_card, text=f"📦 Objetos: {objs_str}", font=ctk.CTkFont(size=11), text_color="#cccccc")
            obj_lbl.pack(anchor="w", pady=(0, 6))

            # Botón para ver foto si existe
            foto_path = ev.get("fotografia")
            if foto_path and os.path.exists(foto_path):
                btn_ver_foto = ctk.CTkButton(inner_card, text="🔍 Ver Fotografía del Evento", width=180, height=26, fg_color="#2c3e50", hover_color="#34495e", command=lambda r=foto_path: self.ver_fotografia(r))
                btn_ver_foto.pack(anchor="w", pady=(2, 0))

    # ====================== MÉTODOS PTZ ONVIF ======================
    def inicializar_conexion_ptz(self):
        if ONVIFCamera is None:
            return False
        ip = self.entry_onvif_ip.get().strip()
        port_str = self.entry_onvif_port.get().strip()
        user = self.entry_onvif_user.get().strip()
        password = self.entry_onvif_pass.get().strip()

        if not ip:
            return False

        port = int(port_str) if port_str.isdigit() else 80
        try:
            self.onvif_client = ONVIFCamera(ip, port, user, password)
            self.media_service = self.onvif_client.create_media_service()
            self.ptz_service = self.onvif_client.create_ptz_service()
            profiles = self.media_service.GetProfiles()
            if profiles:
                self.token_profile = profiles[0].token
                return True
        except Exception as e:
            print(f"Error conectando PTZ: {e}")
        return False

    def mover_ptz_manual(self, pan, tilt):
        threading.Thread(target=lambda: self.mover_ptz_relativo(pan, tilt), daemon=True).start()

    def mover_ptz_relativo(self, pan=0.0, tilt=0.0, zoom=0.0):
        if not self.ptz_service or not self.token_profile:
            return
        try:
            req = self.ptz_service.create_type('ContinuousMove')
            req.ProfileToken = self.token_profile
            req.Velocity = {'PanTilt': {'x': pan, 'y': tilt}, 'Zoom': {'x': zoom}}
            self.ptz_service.ContinuousMove(req)
            
            def detener():
                time.sleep(0.3)
                try:
                    self.ptz_service.Stop({'ProfileToken': self.token_profile})
                except Exception:
                    pass
            threading.Thread(target=detener, daemon=True).start()
        except Exception as e:
            print(f"Error PTZ: {e}")

    def ajustar_giro_por_deteccion(self, bbox_x_centro, frame_ancho):
        if not self.switch_ptz_auto.get():
            return
        ahora = time.time()
        if ahora - self.ultimo_movimiento_ptz < 1.5:
            return

        centro_frame = frame_ancho / 2
        desplazamiento = bbox_x_centro - centro_frame
        
        if abs(desplazamiento) < (frame_ancho * 0.15):
            return

        pan_vel = 0.4 if desplazamiento > 0 else -0.4
        self.ultimo_movimiento_ptz = ahora
        threading.Thread(target=lambda: self.mover_ptz_relativo(pan=pan_vel, tilt=0.0), daemon=True).start()

    # ====================== LÓGICA DE CÁMARA Y EVENTOS ======================
    def cambiar_tipo_camara(self, choice):
        if "RTSP" in choice or "Seguridad" in choice:
            self.label_rtsp.pack(pady=(4, 0), padx=10, anchor="w")
            self.entry_rtsp.pack(pady=2, padx=10)
            self.card_ptz.pack(fill="x", pady=4, padx=4)
        else:
            self.label_rtsp.pack_forget()
            self.entry_rtsp.pack_forget()
            self.card_ptz.pack_forget()

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
                messagebox.showwarning("URL Vacía", "Por favor escribe la URL RTSP.")
                return
            fuente = url
            if self.switch_ptz_auto.get():
                threading.Thread(target=self.inicializar_conexion_ptz, daemon=True).start()
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

        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo conectar a la fuente de video.")
            return

        self.camera_running = True
        self.btn_camara.configure(text="⏹️ Detener Cámara", fg_color="#c0392b", hover_color="#922b21")
        self.status.configure(text="Cámara Activa")

        threading.Thread(target=self._hilo_captura, daemon=True).start()
        threading.Thread(target=self.actualizar_camara, daemon=True).start()

    def detener_camara(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_camara.configure(text="▶️ Iniciar Cámara", fg_color="#e67e22", hover_color="#d35400")
        self.camera_label.configure(image=None, text="Cámara Apagada")
        self.status.configure(text="Cámara Detenida")

    def _hilo_captura(self):
        while self.camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.frame_lock:
                    self.latest_raw_frame = frame
            else:
                time.sleep(0.005)

    def actualizar_camara(self):
        while self.camera_running:
            with self.frame_lock:
                if self.latest_raw_frame is None:
                    time.sleep(0.01)
                    continue
                frame = self.latest_raw_frame.copy()

            if self.es_webcam:
                frame = cv2.flip(frame, 1)
            frame = self.aplicar_zoom(frame)
            h_frame, w_frame = frame.shape[:2]

            boxes_to_draw = []
            if self.yolo_model is not None:
                try:
                    results = self.yolo_model.track(frame, persist=True, verbose=False, conf=0.45, imgsz=320)
                    for r in results:
                        if r.boxes is None: continue
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            nombre = str(self.yolo_model.names[cls_id]).lower()
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            boxes_to_draw.append((nombre, x1, y1, x2, y2, f"{nombre} {conf:.2f}"))
                            if nombre == "person":
                                self.ajustar_giro_por_deteccion((x1 + x2) / 2, w_frame)
                except Exception as e:
                    print(f"Error YOLO: {e}")

            display = frame.copy()
            for nombre, x1, y1, x2, y2, etiqueta in boxes_to_draw:
                color = (0, 220, 100) if nombre == "person" else (255, 160, 0)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, etiqueta, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            self.after(0, lambda r=rgb: self._actualizar_vista_ui(r))
            time.sleep(0.01)

    def _actualizar_vista_ui(self, rgb):
        if not self.winfo_exists(): return
        try:
            img = Image.fromarray(rgb).resize((720, 450), Image.BILINEAR)
            photo = ctk.CTkImage(light_image=img, dark_image=img, size=(720, 450))
            self.camera_label.configure(image=photo, text="")
            self.camera_label.image = photo
        except Exception:
            pass

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
                                    eventos.append(ev)
                                except json.JSONDecodeError:
                                    continue
                except Exception:
                    pass
        return eventos

    def actualizar_estadisticas(self): pass
    def cargar_img1(self): pass
    def cargar_img2(self): pass
    def iniciar_verificacion(self): pass
    
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
            ctk.CTkButton(fila, text="Ver", width=70, command=lambda r=ruta: self.ver_fotografia(r)).pack(side="right", padx=8)

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

    def limpiar_registro(self): self.resultado.delete("1.0", "end")

    def on_closing(self):
        self.camera_running = False
        if self.cap: self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = DeepFaceApp()
    app.mainloop()
