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

    # Regiones clave del rostro
    ceja_izq    = gray[int(h*0.15):int(h*0.30), int(w*0.10):int(w*0.45)]
    ceja_der    = gray[int(h*0.15):int(h*0.30), int(w*0.55):int(w*0.90)]
    ojos        = gray[int(h*0.25):int(h*0.45), int(w*0.10):int(w*0.90)]
    nariz       = gray[int(h*0.40):int(h*0.60), int(w*0.30):int(w*0.70)]
    boca        = gray[int(h*0.60):int(h*0.82), int(w*0.20):int(w*0.80)]
    mejilla_izq = gray[int(h*0.45):int(h*0.70), int(w*0.05):int(w*0.30)]
    mejilla_der = gray[int(h*0.45):int(h*0.70), int(w*0.70):int(w*0.95)]

    # Detectar bordes (activación muscular)
    bordes_boca  = cv2.Canny(boca,  30, 100)
    bordes_ojos  = cv2.Canny(ojos,  30, 100)
    bordes_cejas = cv2.Canny(
        np.hstack([ceja_izq, ceja_der]), 30, 100
    )

    activacion_boca  = np.sum(bordes_boca)  / (boca.size  + 1)
    activacion_ojos  = np.sum(bordes_ojos)  / (ojos.size  + 1)
    activacion_cejas = np.sum(bordes_cejas) / (bordes_cejas.size + 1)

    # Simetría de mejillas (asimetría = tensión)
    simetria = abs(np.mean(mejilla_izq) - np.mean(mejilla_der))

    # Brillo general
    brillo = np.mean(gray) / 255.0

    # Detección de boca abierta (sonrisa amplia / sorpresa)
    _, boca_thresh = cv2.threshold(boca, 80, 255, cv2.THRESH_BINARY_INV)
    zona_oscura_boca = np.sum(boca_thresh) / (boca.size * 255 + 1)

    # Detección de ojos muy abiertos (sorpresa / miedo)
    _, ojos_thresh = cv2.threshold(ojos, 60, 255, cv2.THRESH_BINARY_INV)
    zona_oscura_ojos = np.sum(ojos_thresh) / (ojos.size * 255 + 1)

    # Puntuación por emoción
    scores = {
        'happy':    activacion_boca * 3.0
                  + zona_oscura_boca * 2.5
                  + brillo * 2.0
                  - activacion_cejas * 0.5,

        'sad':      activacion_cejas * 2.5
                  + (1 - brillo) * 2.0
                  + simetria * 0.03
                  - activacion_boca * 1.0,

        'angry':    activacion_cejas * 3.0
                  + simetria * 0.04
                  + (1 - brillo) * 1.5
                  - zona_oscura_boca * 0.5,

        'surprise': zona_oscura_boca * 3.5
                  + zona_oscura_ojos * 2.5
                  + activacion_ojos * 1.5
                  - activacion_cejas * 0.5,

        'fear':     zona_oscura_ojos * 2.5
                  + activacion_cejas * 2.0
                  + (1 - brillo) * 1.5
                  - zona_oscura_boca * 0.3,

        'neutral':  max(0, 2.0
                  - activacion_boca * 1.5
                  - activacion_cejas * 1.5
                  - zona_oscura_boca * 1.0),

        'disgust':  activacion_cejas * 2.0
                  + simetria * 0.05
                  + activacion_boca * 1.0
                  - brillo * 1.0,
    }

    # Normalizar a porcentajes
    total = sum(max(v, 0) for v in scores.values()) + 1e-6
    porcentajes = {k: (max(v, 0) / total) * 100 for k, v in scores.items()}
    emocion = max(porcentajes, key=porcentajes.get)
    return emocion, porcentajes[emocion]

def mostrar_emociones():
    st.title("🎭 Detector de Emociones")
    st.write("Sube una imagen con un rostro para analizar su emoción.")

    archivo = st.file_uploader(
        "Sube tu imagen",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if archivo is not None:
        img_pil = Image.open(archivo)
        fname   = convertir_a_jpg(img_pil)

        img     = cv2.imread(fname)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
        )
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(
                gray_eq, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
            )

        if len(faces) == 0:
            st.warning("⚠️ No se detectó ningún rostro humano en la imagen.")
            st.image(img_rgb, use_column_width=True)
            os.unlink(fname)
            return

        img_h, img_w = img.shape[:2]

        # Tamaño de figura proporcional a la imagen (máx 8)
        ratio  = img_w / img_h
        fig_w  = min(8, 8 * ratio)
        fig_h  = min(8, 8 / ratio)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(img_rgb)
        conteo = []

        for (x, y, w, h) in faces:
            if (w * h) / (img_w * img_h) > 0.80:
                continue

            rostro_recortado   = img_rgb[y:y+h, x:x+w]
            emocion, confianza = analizar_emocion(rostro_recortado)

            ax.add_patch(plt.Rectangle(
                (x, y), w, h, fill=False, color='green', linewidth=2
            ))
            ax.text(
                x, y - 10,
                f"{emocion.upper()} {confianza:.1f}%",
                bbox=dict(facecolor='yellow', alpha=0.7),
                fontsize=10, color='black', fontweight='bold'
            )
            conteo.append(emocion)

        if not conteo:
            st.warning("⚠️ No se detectó ningún rostro humano en la imagen.")
            st.image(img_rgb, use_column_width=True)
            os.unlink(fname)
            return

        ax.axis('off')
        ax.set_title("Analizador de Emociones — OpenCV", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

        st.write("### Resumen Final")
        for emo, cant in Counter(conteo).items():
            st.success(f"✅ {emo.capitalize()}: {cant} rostro(s)")

        os.unlink(fname)
