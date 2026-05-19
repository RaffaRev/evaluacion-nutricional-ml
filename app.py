import streamlit as st
import pandas as pd
import joblib

# 1. Configuración de la estructura de la página
st.set_page_config(page_title="Evaluador Nutricional", layout="centered")

st.title("Sistema de Clasificación de Alimentos (Machine Learning)")
st.write("Examen Final - Despliegue de Modelos de Clasificación Binaria")

# Enlace directo al repositorio de desarrollo en la barra lateral (Modo Lectura)
st.sidebar.header("Documentación del Proyecto")
st.sidebar.write("Acceda al cuaderno original de Google Colab para revisar el Análisis Exploratorio (EDA) y el modelado estadístico:")
# NOTA: Reemplace esta URL por su enlace real de compartir en Colab si lo desea
url_colab = "https://colab.research.google.com/drive/1-5SIoqvwNV4fMvLP7FN2NS9P4gjampz0?usp=sharing"
st.sidebar.markdown(f"[📘 Ver Google Colab (Solo Lectura)]({url_colab})")
st.sidebar.divider()

# 2. Selector del modelo matemático a emplear
opcion_modelo = st.selectbox(
    "Seleccione el algoritmo para la predicción:",
    ("Regresión Logística", "Random Forest")
)

# Mapeo exacto a los nombres de los archivos en su GitHub (con los nombres actuales)
if opcion_modelo == "Regresión Logística":
    archivo_modelo = 'logistic_regression_model.pkl'
else:
    archivo_modelo = 'ramdom_forest_model.pkl'  # Mantiene el nombre exacto de su GitHub

@st.cache_resource
def cargar_modelo(path):
    return joblib.load(path)

try:
    model = cargar_modelo(archivo_modelo)
except Exception as e:
    st.error(f"Error al cargar {archivo_modelo}. Asegúrese de que el archivo esté en la raíz de GitHub.")
    st.stop()

st.divider()

# 3. Interfaz de usuario para la captura de Características (Features)
st.subheader("Parámetros Nutricionales del Alimento")

calories = st.slider("Calorías (Calories)", min_value=0.0, max_value=1000.0, value=250.0, step=1.0)
protein = st.slider("Proteína (Protein) en gramos", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
fat = st.slider("Grasa total (Fat) en gramos", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

# Variables adicionales requeridas por las dimensiones del modelo entrenado
added_sugars = st.slider("Azúcares Añadidos (Added_Sugars_DV %)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
calcium = st.slider("Calcio (Calcium_DV %)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
carbs = st.slider("Carbohidratos (Carbs_DV %)", min_value=0.0, max_value=100.0, value=20.0, step=0.5)
cholesterol = st.slider("Colesterol (Cholesterol_DV %)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
fiber = st.slider("Fibra (Fiber_DV %)", min_value=0.0, max_value=100.0, value=8.0, step=0.5)

# =========================================================================
# 4. CONSTRUCCIÓN DEL DATAFRAME CON EL ORDEN EXACTO DE ENTRENAMIENTO
# =========================================================================
# Los nombres y el orden corresponden estrictamente a la matriz X de entrenamiento
datos_entrada = pd.DataFrame([{
    'Calories': calories,
    'Protein': protein,
    'Fat': fat,
    'Added_Sugars_DV': added_sugars,
    'Calcium_DV': calcium,
    'Carbs_DV': carbs,
    'Cholesterol_DV': cholesterol,
    'Fiber_DV': fiber
}])

st.divider()

# 5. Ejecución del método predict ante la acción del usuario
if st.button("Evaluar Perfil Alimentario"):
    try:
        prediccion = model.predict(datos_entrada)[0]
        
        st.subheader("Resultado del Análisis:")
        if prediccion == 0:
            st.success("🟢 ALIMENTO SALUDABLE: El perfil nutricional se encuentra dentro de los márgenes aceptables.")
        else:
            st.error("🔴 ALIMENTO NO SALUDABLE: Se detectaron excesos o desbalances críticos en los componentes analizados.")
            
    except Exception as error:
        st.error(f"Error en la predicción: {error}. Verifique la correspondencia de las variables en el DataFrame.")
