import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import tempfile
import os

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

@st.cache_resource
def cargar_modelo():
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    return HSEmotionRecognizer(model_name='enet_b0_8_best_afew')

def es_rostro_real(gray_rostro):
    h     = gray_rostro.shape[0]
    mitad = gray_rostro[:h//2, :]
    ojos  = eye_cascade.detectMultiScale(
        mitad, scaleFactor=1.1, minNeighbors=3, minSize=(10,10)
    )
    return len(ojos) >= 1

def convertir_a_jpg(img_pil):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img_pil.convert("RGB").save(f.name, "JPEG")
        return f.name

def mostrar_emociones():
    st.title("🎭 Detector de Emociones")
    st.write("Sube una imagen con un rostro para analizar su emoción.")

    archivo = st.file_uploader(
        "Sube tu imagen",
        type=["jpg","jpeg","png","webp"]
    )

    if archivo is not None:
        fer = cargar_modelo()

        img_pil = Image.open(archivo)
        fname   = convertir_a_jpg(img_pil)
        img     = cv2.imread(fname)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.1,
            minNeighbors=6, minSize=(60,60)
        )
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(
                gray_eq, scaleFactor=1.05,
                minNeighbors=3, minSize=(40,40)
            )

        if len(faces) == 0:
            st.warning("⚠️ No se detectó ningún rostro humano en la imagen.")
            st.image(img_rgb, use_column_width=True)
            os.unlink(fname)
            return

        img_h, img_w = img.shape[:2]

        if len(faces) > 1:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            faces = [faces[0]]

        ratio = img_w / img_h
        fig_w = min(7, 7 * ratio)
        fig_h = min(7, 7 / ratio)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(img_rgb)
        conteo = []

        for (x, y, w, h) in faces:
            if (w * h) / (img_w * img_h) > 0.80:
                continue
            gray_rostro = gray[y:y+h, x:x+w]
            if not es_rostro_real(gray_rostro):
                continue

            rostro_rgb = img_rgb[y:y+h, x:x+w]

            with st.spinner("Analizando emoción..."):
                emocion, scores = fer.predict_emotions(rostro_rgb, logits=False)
                confianza = float(np.max(scores)) * 100

            ax.add_patch(plt.Rectangle(
                (x, y), w, h,
                fill=False, color='green', linewidth=2
            ))
            ax.text(
                x, y - 10,
                f"{emocion.upper()} {confianza:.1f}%",
                bbox=dict(facecolor='yellow', alpha=0.8),
                fontsize=10, color='black', fontweight='bold'
            )
            conteo.append(emocion)

        if not conteo:
            st.warning("⚠️ No se detectó ningún rostro humano en la imagen.")
            st.image(img_rgb, use_column_width=True)
            os.unlink(fname)
            return

        ax.axis('off')
        ax.set_title("Analizador de Emociones — HSEmotion", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

        st.write("### Resumen Final")
        for emo, cant in Counter(conteo).items():
            st.success(f"✅ {emo.capitalize()}: {cant} rostro(s)")

        os.unlink(fname)
