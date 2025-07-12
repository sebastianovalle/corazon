# APP predicción problema cardiaco
'''


'''

import streamlit as st
import joblib
import pandas as pd
from PIL import Image

# Cargar el modelo y el scaler
try:
    svc_model_loaded = joblib.load('svc_model.jb')
    scaler_loaded = joblib.load('scaler.jb')
except FileNotFoundError:
    st.error("Archivos de modelo (svc_model.jb, scaler.jb) no encontrados. Asegúrese de que estén en el mismo directorio que la aplicación de Streamlit.")
    st.stop() # Detiene la ejecución si los archivos no se encuentran

# Cargar imágenes
try:
    cabezote_img = Image.open('fondo.jpg')
    nosufre_img = Image.open('nosufre.jpg')
    sisufre_img = Image.open('sisufre.jpg')
except FileNotFoundError:
    st.warning("Archivos de imagen (cabezote.jpg, nosufre.jpg, sisufre.jpg) no encontrados.")
    cabezote_img = None
    nosufre_img = None
    sisufre_img = None


# Título y descripción
if cabezote_img:
    st.image(cabezote_img, use_container_width=True)
st.title("Modelo IA para predicción de problemas cardiacos")

st.write("""
Este modelo utiliza un algoritmo de Support Vector Classifier (SVC) entrenado con datos de pacientes (edad y colesterol) para predecir la probabilidad de sufrir problemas cardiacos.
Los datos de entrada (Edad y Colesterol) se escalan utilizando un MinMaxScaler antes de ser introducidos al modelo para asegurar un mejor rendimiento.
El modelo predice '0' si se espera que el paciente no sufra problemas cardiacos y '1' si se espera que sí sufra.
""")

# Sidebar para la entrada del usuario
st.sidebar.header("Parámetros del Paciente")

edad = st.sidebar.slider(
    "Edad:",
    min_value=20,
    max_value=80,
    value=20,
    step=1
)

colesterol = st.sidebar.slider(
    "Colesterol:",
    min_value=120,
    max_value=600,
    value=200,
    step=10
)

# Preparar los datos de entrada para la predicción
input_data = pd.DataFrame({'edad': [edad], 'colesterol': [colesterol]})

# Escalar los datos de entrada
input_data_scaled = scaler_loaded.transform(input_data)
input_data_scaled = pd.DataFrame(input_data_scaled, columns=['edad', 'colesterol'])

# Realizar la predicción
prediction = svc_model_loaded.predict(input_data_scaled)

# Mostrar los resultados
st.subheader("Resultado de la predicción:")

if prediction[0] == 0:
    st.markdown("<h2 style='color: green;'>0: No sufrirá del corazón 😊</h2>", unsafe_allow_html=True)
    if nosufre_img:
        st.image(nosufre_img, use_column_width=True)
else:
    st.markdown("<h2 style='color: red;'>1: Sufrirá del corazón 😥</h2>", unsafe_allow_html=True)
    if sisufre_img:
        st.image(sisufre_img, use_column_width=True)

st.write("") # Espacio
st.write("Elaborado por: Sebastian Ovalle")
