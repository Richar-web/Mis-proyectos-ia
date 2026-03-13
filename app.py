import streamlit as st

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Portal de Inteligencia Artificial",
    page_icon="🤖",
    layout="wide"
)

# 2. CARGAR EL CSS PERSONALIZADO
try:
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass

# 3. MENÚ LATERAL (Navegación)
with st.sidebar:
    st.markdown('<h2 style="color:#00ffcc;">🤖 Panel de Control</h2>', unsafe_allow_html=True)
    opcion = st.radio(
        "Selecciona un Proyecto:",
        ["🏠 Inicio", "🧠 Detector de Emociones", "📦 Detector de Objetos"]
    )
    st.write("---")
    st.info("Desarrollado con TensorFlow y PyTorch")

# 4. LÓGICA DE NAVEGACIÓN
if opcion == "🏠 Inicio":
    st.markdown('<h1 class="titulo-neon">BIENVENIDO AL PORTAL IA</h1>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="card-html">
            <h3>Centro de Proyectos de Inteligencia Artificial</h3>
            <p>Explora las capacidades de la IA a través de estos dos módulos especializados:</p>
            <ul>
                <li><b>Módulo de Emociones:</b> Análisis facial mediante Redes Neuronales Convolucionales (DeepFace).</li>
                <li><b>Módulo de Objetos:</b> Detección múltiple con arquitectura Faster R-CNN (PyTorch).</li>
            </ul>
            <p style="color:#00ffcc;"><i>Selecciona una opción en el menú de la izquierda para comenzar.</i></p>
        </div>
    """, unsafe_allow_html=True)

elif opcion == "🧠 Detector de Emociones":
    import emociones
    emociones.mostrar_emociones()

elif opcion == "📦 Detector de Objetos":
    import objetos
    objetos.mostrar_objetos()
