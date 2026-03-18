import streamlit as st  # Importa Streamlit, el framework para crear la interfaz web interactiva
import torch  # Importa PyTorch, la librería principal de deep learning
import torchvision  # Importa TorchVision, que contiene modelos y utilidades para visión computacional
from torchvision.models.detection import fasterrcnn_resnet50_fpn  # Importa el modelo Faster R-CNN con backbone ResNet-50 + FPN (red de pirámide de características)
from torchvision.transforms import functional as F  # Importa funciones de transformación de imágenes (la llamamos F para abreviar)
from PIL import Image  # Importa PIL (Pillow) para abrir y manipular imágenes
import matplotlib.pyplot as plt  # Importa matplotlib para dibujar la imagen con los recuadros de detección
from collections import Counter  # Importa Counter para contar cuántas veces aparece cada objeto detectado
import tempfile, os  # Importa módulos para archivos temporales y operaciones del sistema operativo (aunque no se usan directamente en este código)

# Lista completa de las 91 categorías del dataset COCO (Common Objects in Context)
# El índice de cada elemento corresponde al ID de clase que devuelve el modelo
# '__background__' es la clase 0 (el fondo, no es un objeto), 'N/A' son clases reservadas sin uso
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
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

@st.cache_resource  # Decorador de Streamlit: guarda el modelo en caché para no recargarlo en cada interacción del usuario
def cargar_modelo():  # Define la función que carga el modelo de detección
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT  # Carga los pesos preentrenados por defecto (entrenados en COCO)
    model   = fasterrcnn_resnet50_fpn(weights=weights)  # Crea el modelo Faster R-CNN con esos pesos preentrenados
    model.eval()  # Pone el modelo en modo evaluación (desactiva dropout y batch norm para inferencia, no para entrenamiento)
    return model  # Devuelve el modelo listo para usar

def mostrar_objetos():  # Define la función principal que contiene toda la lógica de la aplicación
    st.title("📦 Detector de Objetos")  # Muestra el título principal en la interfaz web
    st.write("Sube una imagen para detectar los objetos que contiene.")  # Muestra un mensaje de instrucción al usuario

    archivo = st.file_uploader(  # Crea un widget de carga de archivos en la interfaz
        "Sube tu imagen",  # Etiqueta que ve el usuario sobre el botón de carga
        type=["jpg","jpeg","png","webp"]  # Restringe los formatos aceptados a estos tipos de imagen
    )

    if archivo is not None:  # Solo ejecuta el bloque siguiente si el usuario subió un archivo
        image = Image.open(archivo).convert("RGB")  # Abre la imagen subida y la convierte a RGB (por si tiene canal alpha o es escala de grises)

        with st.spinner("Cargando modelo y detectando objetos..."):  # Muestra un spinner (animación de carga) mientras procesa
            model      = cargar_modelo()  # Obtiene el modelo (de caché si ya fue cargado antes)
            img_tensor = F.to_tensor(image)  # Convierte la imagen PIL a un tensor de PyTorch con valores entre 0 y 1
            with torch.no_grad():  # Desactiva el cálculo de gradientes (ahorra memoria y acelera la inferencia)
                pred = model([img_tensor])  # Pasa la imagen al modelo; devuelve predicciones (cajas, scores, etiquetas)

        boxes  = pred[0]['boxes']   # Extrae las coordenadas de los bounding boxes detectados [x1, y1, x2, y2]
        scores = pred[0]['scores']  # Extrae las puntuaciones de confianza de cada detección (entre 0 y 1)
        labels = pred[0]['labels']  # Extrae los IDs de clase de cada detección

        keep          = torchvision.ops.batched_nms(boxes, scores, labels, iou_threshold=0.3)  # Aplica NMS (Non-Maximum Suppression) por clase: elimina cajas redundantes que se solapan más del 30% dentro de la misma clase
        final_keep    = torchvision.ops.nms(boxes[keep], scores[keep], iou_threshold=0.3)  # Aplica NMS global sobre las cajas ya filtradas: elimina solapamientos entre distintas clases también
        indices_finales = keep[final_keep]  # Combina los dos filtros para obtener los índices definitivos de las detecciones válidas

        fig, ax = plt.subplots(figsize=(12, 8))  # Crea una figura de matplotlib de 12x8 pulgadas para visualizar la imagen
        ax.imshow(image)  # Dibuja la imagen original como fondo del gráfico

        conteo_objetos = []  # Lista vacía donde se irán guardando los nombres de objetos detectados (para el resumen)

        objetos_dudosos = ['dining table', 'tie', 'chair']  # Lista de objetos que tienden a dar falsos positivos: se les exige mayor confianza

        for i in indices_finales:  # Itera sobre cada índice de detección válida
            score      = scores[i].item()       # Obtiene la confianza de esa detección como número Python (no tensor)
            label_id   = labels[i].item()       # Obtiene el ID numérico de la clase detectada
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]  # Convierte el ID al nombre legible usando la lista COCO

            umbral     = 0.85 if label_name in objetos_dudosos else 0.50  # Asigna umbral de confianza: 85% para objetos dudosos, 50% para el resto

            if score > umbral:  # Solo procesa la detección si supera el umbral de confianza correspondiente
                box = boxes[i].detach().numpy()  # Extrae la caja del tensor y la convierte a array NumPy (detach separa del grafo de gradientes)
                x1, y1, x2, y2 = box  # Desempaqueta las coordenadas: esquina superior izquierda (x1,y1) y esquina inferior derecha (x2,y2)

                ax.add_patch(plt.Rectangle(  # Dibuja un rectángulo sobre la imagen en la posición de la caja detectada
                    (x1, y1), x2-x1, y2-y1,  # Posición (x1,y1) y tamaño (ancho = x2-x1, alto = y2-y1)
                    fill=False, color='red', linewidth=2  # Sin relleno, borde rojo de 2px de grosor
                ))

                ax.text(  # Escribe el texto con el nombre del objeto y su confianza sobre la caja
                    x1, y1,  # Posición del texto: esquina superior izquierda de la caja
                    f"{label_name} {score:.2f}",  # Texto: nombre del objeto + confianza con 2 decimales
                    bbox=dict(facecolor='yellow', alpha=0.5),  # Fondo amarillo semitransparente detrás del texto
                    fontsize=10, color='black', fontweight='bold'  # Fuente negra en negrita de tamaño 10
                )

                conteo_objetos.append(label_name)  # Agrega el nombre del objeto a la lista para el resumen final

        ax.axis('off')  # Oculta los ejes del gráfico (los números de coordenadas) para una visualización más limpia
        plt.tight_layout()  # Ajusta automáticamente los márgenes del gráfico para que no haya recortes
        st.pyplot(fig)  # Muestra el gráfico con la imagen y las detecciones dentro de la interfaz de Streamlit

        st.write("### Resumen de Detección Final")  # Muestra el subtítulo del resumen en la interfaz
        resumen = Counter(conteo_objetos)  # Cuenta cuántas veces aparece cada nombre de objeto en la lista

        if not resumen:  # Si no se detectó ningún objeto (el diccionario está vacío)
            st.warning("No se detectaron objetos con la confianza suficiente.")  # Muestra una advertencia en amarillo
        else:  # Si sí hubo detecciones
            for obj, cant in resumen.items():  # Itera sobre cada par (nombre_objeto, cantidad)
                st.success(f"✅ {obj.capitalize()}: {cant}")  # Muestra cada objeto con su cantidad en un cuadro verde de éxito
