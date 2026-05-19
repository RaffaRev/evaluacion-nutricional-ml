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

# 3. Interfaz de usuario para la captura de Características (Todas las columnas de X)
st.subheader("Parámetros Nutricionales del Alimento")
st.write("Ajuste los valores según la etiqueta nutricional del producto para proceder con la clasificación.")

# Bloque 1: Macronutrientes y Calorías
calories = st.slider("Calorías (Calories)", min_value=0.0, max_value=1000.0, value=250.0, step=1.0)
total_fat_dv = st.slider("Grasa Total (Total_Fat_DV %)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
sat_fat_dv = st.slider("Grasa Saturada (Sat_Fat_DV %)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
trans_fat = st.slider("Grasas Trans (Trans_Fat) en gramos", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
proteins_g = st.slider("Proteínas (Proteins_G) en gramos", min_value=0.0, max_value=100.0, value=15.0, step=0.1)

# Bloque 2: Carbohidratos y Azúcares
carbs_dv = st.slider("Carbohidratos (Carbs_DV %)", min_value=0.0, max_value=100.0, value=30.0, step=0.5)
fiber_dv = st.slider("Fibra Alimentaria (Fiber_DV %)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
total_sugars_g = st.slider("Azúcares Totales (Total_Sugars_G) en gramos", min_value=0.0, max_value=150.0, value=12.0, step=0.5)
added_sugars_dv = st.slider("Azúcares Añadidos (Added_Sugars_DV %)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)

# Bloque 3: Minerales, Vitaminas y Otros Componentes
cholesterol_dv = st.slider("Colesterol (Cholesterol_DV %)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
sodium_dv = st.slider("Sodio (Sodium_DV %)", min_value=0.0, max_value=100.0, value=8.0, step=0.5)
vitamin_d_dv = st.slider("Vitamina D (Vitamin_D_DV %)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
calcium_dv = st.slider("Calcio (Calcium_DV %)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
iron_dv = st.slider("Hierro (Iron_DV %)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
potassium_dv = st.slider("Potasio (Potassium_DV %)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)

# =========================================================================
# 4. CONSTRUCCIÓN DEL DATAFRAME CON EL ORDEN EXACTO DE ENTRENAMIENTO
# =========================================================================
# Se excluyen 'Food_Id' y 'Healthiness' de forma estricta.
datos_entrada = pd.DataFrame([{
    'Calories': calories,
    'Total_Fat_DV': total_fat_dv,
    'Sat_Fat_DV': sat_fat_dv,
    'Trans_Fat': trans_fat,
    'Cholesterol_DV': cholesterol_dv,
    'Sodium_DV': sodium_dv,
    'Carbs_DV': carbs_dv,
    'Fiber_DV': fiber_dv,
    'Total_Sugars_G': total_sugars_g,
    'Added_Sugars_DV': added_sugars_dv,
    'Proteins_G': proteins_g,
    'Vitamin_D_DV': vitamin_d_dv,
    'Calcium_DV': calcium_dv,
    'Iron_DV': iron_dv,
    'Potassium_DV': potassium_dv
}])

st.divider()

# 5. Ejecución del método predict ante la acción del usuario
if st.button("Evaluar Perfil Alimentario"):
    try:
        prediccion = model.predict(datos_entrada)[0]
        
        st.subheader("Resultado del Análisis:")
        if prediccion == 0:
            st.success("🟢 ALIMENTO SALUDABLE: El perfil nutritional se encuentra dentro de los márgenes aceptables.")
        else:
            st.error("🔴 ALIMENTO NO SALUDABLE: Se detectaron excesos o desbalances críticos en los componentes analizados.")
            
    except Exception as error:
        st.error(f"Error en la predicción: {error}. Verifique la correspondencia de las variables en el DataFrame.")
