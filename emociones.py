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

def convertir_a_jpg(img_pil):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img_pil.convert("RGB").save(f.name, "JPEG")
        return f.name

def analizar_emocion(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    rostro_superior = gray[:h//2, :]
    rostro_inferior = gray[h//2:, :]

    brillo_promedio = np.mean(gray)
    contraste       = np.std(gray)

    region_ojos  = gray[int(h*0.2):int(h*0.45), int(w*0.1):int(w*0.9)]
    region_boca  = gray[int(h*0.6):int(h*0.85), int(w*0.2):int(w*0.8)]
    region_cenos = gray[int(h*0.15):int(h*0.35), int(w*0.2):int(w*0.8)]

    variacion_boca  = np.std(region_boca)
    variacion_ojos  = np.std(region_ojos)
    variacion_cenos = np.std(region_cenos)
    diferencia_v    = np.mean(rostro_superior) - np.mean(rostro_inferior)

    scores = {
        'happy':   variacion_boca * 1.5 + (brillo_promedio / 255) * 30,
        'sad':     (1 - brillo_promedio / 255) * 40 + variacion_cenos * 0.8,
        'angry':   variacion_cenos * 1.4 + contraste * 0.5,
        'surprise':variacion_ojos * 1.6 + abs(diferencia_v) * 0.4,
        'fear':    contraste * 0.9 + variacion_cenos * 0.7,
        'neutral': 100 - contraste * 0.5 - variacion_boca * 0.3,
        'disgust': variacion_cenos * 1.1 + variacion_boca * 0.6,
    }

    total    = sum(scores.values())
    porcentajes = {k: (v / total) * 100 for k, v in scores.items()}
    emocion  = max(porcentajes, key=porcentajes.get)
    return emocion, porcentajes[emocion]

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

        img_h, img_w = img.shape[:2]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(img_rgb)
        conteo = []

        for (x, y, w, h) in faces:
            if (w * h) / (img_w * img_h) > 0.80:
                continue

            rostro_recortado = img_rgb[y:y+h, x:x+w]
            emocion, confianza = analizar_emocion(rostro_recortado)

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
        ax.set_title("Analizador de Emociones — OpenCV", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)

        st.write("### Resumen Final")
        for emo, cant in Counter(conteo).items():
            st.success(f"✅ {emo.capitalize()}: {cant} rostro(s)")

        os.unlink(fname)
