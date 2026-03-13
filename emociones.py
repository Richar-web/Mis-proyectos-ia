import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import tempfile
import os

def mostrar_emociones():
    st.title("🎭 Detector de Emociones")
    st.write("Sube una imagen con un rostro para analizar su emoción.")

    archivo = st.file_uploader(
        "Sube tu imagen",
        type=["jpg","jpeg","png","webp"]
    )

    if archivo is not None:
        from deepface import DeepFace

        img_pil = Image.open(archivo)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img_pil.convert("RGB").save(f.name, "JPEG")
            fname = f.name

        img = cv2.imread(fname)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            resultados = DeepFace.analyze(
                img_path=fname,
                actions=['emotion'],
                detector_backend='opencv',
                enforce_detection=False
            )

            if not isinstance(resultados, list):
                resultados = [resultados]

            img_h, img_w = img.shape[:2]
            ratio = img_w / img_h
            fig_w = min(7, 7 * ratio)
            fig_h = min(7, 7 / ratio)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.imshow(img_rgb)
            conteo = []

            for r in resultados:
                emocion = r['dominant_emotion']
                confianza = r['emotion'][emocion]
                region = r.get('region', {})
                x = region.get('x', 0)
                y = region.get('y', 0)
                w = region.get('w', img_w)
                h = region.get('h', img_h)

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

            ax.axis('off')
            ax.set_title("Analizador de Emociones — DeepFace", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)

            st.write("### Resumen Final")
            for emo, cant in Counter(conteo).items():
                st.success(f"✅ {emo.capitalize()}: {cant} rostro(s)")

        except Exception as e:
            st.error(f"Error al analizar: {e}")
            st.image(img_rgb, use_column_width=True)

        os.unlink(fname)
