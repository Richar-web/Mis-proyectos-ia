import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import tempfile
import os
from fer import FER

detector = FER(mtcnn=False)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

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
        img_pil = Image.open(archivo)
        fname   = convertir_a_jpg(img_pil)

        img     = cv2.imread(fname)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        # Verificar que haya rostro
        faces = face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.1, minNeighbors=6, minSize=(60,60)
        )
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(
                gray_eq, scaleFactor=1.05, minNeighbors=3, minSize=(40,40)
            )

        if len(faces) == 0:
            st.warning("⚠️ No se detectó ningún rostro humano en la imagen.")
            st.image(img_rgb)
            os.unlink(fname)
            return

        with st.spinner("Analizando emoción..."):
            resultados = detector.detect_emotions(img_rgb)

        img_h, img_w = img.shape[:2]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(img_rgb)
        conteo = []

        if not resultados:
            st.warning("⚠️ No se detectó ningún rostro humano en la imagen.")
            st.image(img_rgb)
            os.unlink(fname)
            return

        for result in resultados:
            x, y, w, h = result['box']

            if (w * h) / (img_w * img_h) > 0.80:
                continue

            emociones_dict = result['emotions']
            emocion        = max(emociones_dict, key=emociones_dict.get)
            confianza      = emociones_dict[emocion] * 100

            ax.add_patch(plt.Rectangle(
                (x, y), w, h, fill=False, color='green', linewidth=2
            ))
            ax.text(
                x, y - 10,
                f"{emocion.upper()} {confianza:.1f}%",
                bbox=dict(facecolor='yellow', alpha=0.7),
                fontsize=11, color='black', fontweight='bold'
            )
            conteo.append(emocion)

        if not conteo:
            st.warning("⚠️ No se detectó ningún rostro humano en la imagen.")
            st.image(img_rgb)
            os.unlink(fname)
            return

        ax.axis('off')
        ax.set_title("Analizador de Emociones — TensorFlow + FER", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)

        st.write("### Resumen Final")
        for emo, cant in Counter(conteo).items():
            st.success(f"✅ {emo.capitalize()}: {cant} rostro(s)")

        os.unlink(fname)
