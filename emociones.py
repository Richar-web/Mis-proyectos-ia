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

def convertir_a_jpg(img_pil):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img_pil.convert("RGB").save(f.name, "JPEG")
        return f.name

def es_rostro_real(gray_rostro):
    """Verifica que el rostro tenga ojos detectables"""
    h, w = gray_rostro.shape
    mitad_superior = gray_rostro[:h//2, :]
    ojos = eye_cascade.detectMultiScale(
        mitad_superior, scaleFactor=1.1,
        minNeighbors=3, minSize=(15, 15)
    )
    return len(ojos) >= 1

def analizar_emocion(face_rgb):
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Aplicar CLAHE para normalizar iluminación
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    gray  = clahe.apply(gray)

    # Regiones del rostro
    ceja_izq = gray[int(h*0.12):int(h*0.28), int(w*0.08):int(w*0.45)]
    ceja_der = gray[int(h*0.12):int(h*0.28), int(w*0.55):int(w*0.92)]
    ojos     = gray[int(h*0.22):int(h*0.42), int(w*0.08):int(w*0.92)]
    boca     = gray[int(h*0.58):int(h*0.85), int(w*0.18):int(w*0.82)]
    frente   = gray[int(h*0.02):int(h*0.15), int(w*0.20):int(w*0.80)]

    # Bordes (tensión muscular)
    b_cejas = cv2.Canny(np.hstack([ceja_izq, ceja_der]), 40, 120)
    b_boca  = cv2.Canny(boca, 40, 120)
    b_ojos  = cv2.Canny(ojos, 40, 120)

    act_cejas = np.sum(b_cejas) / (b_cejas.size + 1) * 100
    act_boca  = np.sum(b_boca)  / (b_boca.size  + 1) * 100
    act_ojos  = np.sum(b_ojos)  / (b_ojos.size  + 1) * 100

    # Boca abierta = zona oscura entre labios
    _, boca_bin = cv2.threshold(boca, 70, 255, cv2.THRESH_BINARY_INV)
    apertura_boca = np.sum(boca_bin) / (boca.size * 255 + 1)

    # Ojos muy abiertos
    _, ojos_bin = cv2.threshold(ojos, 55, 255, cv2.THRESH_BINARY_INV)
    apertura_ojos = np.sum(ojos_bin) / (ojos.size * 255 + 1)

    # Brillo general del rostro
    brillo = np.mean(gray) / 255.0

    # Arrugas en frente (enojo / sorpresa)
    b_frente  = cv2.Canny(frente, 30, 90)
    act_frente = np.sum(b_frente) / (b_frente.size + 1) * 100

    # Comisuras boca (sonrisa = más ancha que alta)
    boca_h, boca_w = boca.shape
    ratio_boca = boca_w / (boca_h + 1)

    scores = {
        'happy':    apertura_boca * 5.0
                  + ratio_boca * 1.2
                  + brillo * 3.0
                  + act_boca * 0.8
                  - act_cejas * 0.8,

        'sad':      act_cejas * 2.5
                  + (1.0 - brillo) * 4.0
                  + (1.0 - apertura_boca) * 1.5
                  - apertura_ojos * 1.0,

        'angry':    act_cejas * 3.5
                  + act_frente * 1.5
                  + (1.0 - brillo) * 2.0
                  - apertura_boca * 2.0,

        'surprise': apertura_boca * 4.0
                  + apertura_ojos * 4.0
                  + act_frente * 1.0
                  - act_cejas * 0.5,

        'fear':     apertura_ojos * 3.5
                  + act_cejas * 2.0
                  + (1.0 - brillo) * 2.0
                  - apertura_boca * 0.5,

        'neutral':  max(0.0, 3.0
                  - act_cejas * 1.0
                  - apertura_boca * 3.0
                  - act_frente * 0.5),

        'disgust':  act_cejas * 2.0
                  + act_frente * 2.0
                  + (1.0 - apertura_boca) * 1.5
                  - brillo * 1.0,
    }

    total = sum(max(v, 0) for v in scores.values()) + 1e-6
    pct   = {k: (max(v, 0) / total) * 100 for k, v in scores.items()}
    emocion = max(pct, key=pct.get)
    return emocion, pct[emocion]

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

        # Detectar rostros
        faces = face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.1,
            minNeighbors=7, minSize=(80, 80)
        )
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(
                gray_eq, scaleFactor=1.05,
                minNeighbors=5, minSize=(60, 60)
            )

        if len(faces) == 0:
            st.warning("⚠️ No se detectó ningún rostro humano en la imagen.")
            st.image(img_rgb, use_column_width=True)
            os.unlink(fname)
            return

        img_h, img_w = img.shape[:2]

        # Filtrar: quedarse solo con el rostro más grande si hay varios
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
            # Ignorar si cubre >80% de la imagen
            if (w * h) / (img_w * img_h) > 0.80:
                continue

            gray_rostro = gray[y:y+h, x:x+w]

            # Verificar que tenga ojos (descarta cuello/pecho)
            if not es_rostro_real(gray_rostro):
                continue

            rostro_rgb         = img_rgb[y:y+h, x:x+w]
            emocion, confianza = analizar_emocion(rostro_rgb)

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
        ax.set_title("Analizador de Emociones — OpenCV", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

        st.write("### Resumen Final")
        for emo, cant in Counter(conteo).items():
            st.success(f"✅ {emo.capitalize()}: {cant} rostro(s)")

        os.unlink(fname)
