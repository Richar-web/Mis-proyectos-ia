import streamlit as st
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# =====================================
# FUNCIÓN PARA LA INTERFAZ WEB
# =====================================
def mostrar_objetos():
    st.markdown('<h2 style="color:#00ffcc; text-align:center;">📦 Detector de Objetos (IA)</h2>', unsafe_allow_html=True)

    # 1. Categorías COCO (Tus categorías intactas)
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # 2. Cargar modelo (Cache para que la web no sea lenta)
    @st.cache_resource
    def cargar_modelo():
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        model.eval()
        return model

    model = cargar_modelo()

    # 3. Función de procesamiento (Tu lógica de doble filtro intacta)
    def procesar_y_dibujar(im, pred):
        boxes = pred[0]['boxes']
        scores = pred[0]['scores']
        labels = pred[0]['labels']

        # PASO A: Limpiar duplicados de la MISMA clase
        keep = torchvision.ops.batched_nms(boxes, scores, labels, iou_threshold=0.3)

        # PASO B: Limpiar duplicados de CLASES DISTINTAS
        final_keep_idx = torchvision.ops.nms(boxes[keep], scores[keep], iou_threshold=0.3)
        indices_finales = keep[final_keep_idx]

        fig, ax = plt.subplots(figsize=(12,8))
        ax.imshow(im)
        conteo_objetos = []

        for i in indices_finales:
            score = scores[i].item()
            label_id = labels[i].item()
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]

            # Tus Umbrales de confianza personalizados
            objetos_dudosos = ['dining table', 'tie', 'chair']
            umbral_minimo = 0.85 if label_name in objetos_dudosos else 0.50

            if score > umbral_minimo:
                box = boxes[i].detach().numpy()
                x1, y1, x2, y2 = box

                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, color='red', linewidth=2))

                ax.text(x1, y1, f"{label_name} {score:.2f}",
                        bbox=dict(facecolor='yellow', alpha=0.5),
                        fontsize=10, color='black', fontweight='bold')

                conteo_objetos.append(label_name)

        ax.axis('off')
        st.pyplot(fig)

        # Tu resumen final con diseño de tarjeta
        st.markdown('<div class="card-html">', unsafe_allow_html=True)
        st.subheader("📊 RESUMEN DE DETECCIÓN FINAL")
        resumen = Counter(conteo_objetos)
        if not resumen:
            st.write("No se detectaron objetos con la confianza suficiente.")
        for obj, cant in resumen.items():
            st.write(f"✅ **{obj.capitalize()}**: {cant}")
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. Subida de imagen
    archivo = st.file_uploader("Sube una imagen para detectar objetos", type=["jpg", "jpeg", "png"])

    if archivo is not None:
        image = Image.open(archivo).convert("RGB")
        img_tensor = F.to_tensor(image)

        with st.spinner('IA detectando objetos...'):
            with torch.no_grad():
                pred = model([img_tensor])
            procesar_y_dibujar(image, pred)
