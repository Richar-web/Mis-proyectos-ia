import streamlit as st
import tensorflow as tf
from deepface import DeepFace
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import os

# =====================================
# FUNCIÓN PARA LA INTERFAZ WEB
# =====================================
def mostrar_emociones():
    # Usamos la clase de CSS que creamos en styles.css
    st.markdown('<h2 style="color:#00ffcc; text-align:center;">🧠 Analizador de Emociones</h2>', unsafe_allow_html=True)
    
    # 3. DETECTOR DE ROSTROS
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # 4. CONVERTIR CUALQUIER FORMATO A JPG (Tu lógica original)
    def convertir_a_jpg(image_file):
        img = Image.open(image_file).convert("RGB")
        nombre_jpg = "temp_emocion.jpg"
        img.save(nombre_jpg, "JPEG")
        return nombre_jpg

    # 5. FUNCIÓN PRINCIPAL (Adaptada para mostrar resultados en Streamlit)
    def detectar_emociones_web(fname):
        # Mantenemos toda tu lógica de pre-procesamiento
        img = cv2.imread(fname)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        # Tu primer intento de detección
        faces = face_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(60, 60)
        )

        # Tu segundo intento si falla el primero
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(
                gray_eq,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(40, 40)
            )

        if len(faces) == 0:
            st.warning("⚠️ No se detectó ningún rostro humano en la imagen.")
            st.image(img_rgb, caption="Intenta con otra foto")
            return

        # Análisis DeepFace (Tus parámetros exactos)
        with st.spinner('Analizando expresiones...'):
            resultados = DeepFace.analyze(
                img_path=fname,
                actions=['emotion'],
                detector_backend='ssd',
                enforce_detection=False
            )

        img_h, img_w = img.shape[:2]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img_rgb)
        conteo = []

        for i, result in enumerate(resultados):
            region = result['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']

            # Tu validación del 80% (Sin cambios)
            porcentaje_imagen = (w * h) / (img_w * img_h)
            if porcentaje_imagen > 0.80:
                st.error("⚠️ El rostro detectado es demasiado grande o inválido.")
                return

            emocion = result['dominant_emotion']
            confianza = result['emotion'][emocion]

            # Dibujo de rectángulos (Tus colores y estilos)
            ax.add_patch(plt.Rectangle(
                (x, y), w, h,
                fill=False, color='green', linewidth=2
            ))
            ax.text(
                x, y - 10,
                f"{emocion.upper()} {confianza:.1f}%",
                bbox=dict(facecolor='yellow', alpha=0.7),
                fontsize=11, color='black', fontweight='bold'
            )
            conteo.append(emocion)

        if not conteo:
            st.warning("⚠️ No se detectó ningún rostro válido.")
            return

        ax.axis('off')
        # Mostramos tu gráfico en la web
        st.pyplot(fig)

        # Tu resumen final pero con diseño de tarjeta
        st.markdown('<div class="card-html">', unsafe_allow_html=True)
        st.subheader("--- RESUMEN FINAL ---")
        for emo, cant in Counter(conteo).items():
            st.write(f"✅ **{emo.capitalize()}**: {cant} rostro(s)")
        st.markdown('</div>', unsafe_allow_html=True)

    # 6. SUBIR IMAGEN (Reemplaza files.upload)
    uploaded_file = st.file_uploader("Sube tu imagen aquí:", type=["jpg", "png", "jpeg", "webp"])

    if uploaded_file is not None:
        path_temporal = convertir_a_jpg(uploaded_file)
        detectar_emociones_web(path_temporal)
        # Limpieza para no llenar el servidor
        if os.path.exists(path_temporal):
            os.remove(path_temporal)
