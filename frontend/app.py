"""Interfaz alternativa en Streamlit conectada al backend FastAPI."""
import os,requests,streamlit as st
st.set_page_config(page_title='Tipologías de Steam',layout='wide')
URL=os.getenv('STEAM_API_URL','http://127.0.0.1:8000')
st.title('Mapa de tipologías de videojuegos de Steam')
st.caption('Prototipo de clasificación a partir de tags · No representa validación humana de grupos')
try:
    data=requests.get(URL+'/clusters',timeout=15);data.raise_for_status();clusters=data.json()
except Exception as ex:
    st.error('La API no está disponible o no se ha entrenado el modelo. Ejecuta el proceso documentado en README.');st.stop()
st.subheader('Explorar clusters')
cl=st.selectbox('Cluster',list(range(len(clusters))))
st.write(clusters[cl]['top_tags'])
st.subheader('Ubicar un nuevo juego según sus tags')
try:
    tags=sorted(requests.get(URL+'/tags',timeout=10).json()['tags'])
except Exception:
    tags=sorted({row['tag'] for c in clusters for row in c['top_tags']})
st.caption('Selecciona hasta 20 etiquetas del vocabulario entrenado (150 tags) para consultar la tipología más cercana.')
choice=st.multiselect('Selecciona hasta 20 tags',tags,max_selections=20)
if st.button('Clasificar',disabled=not choice):
    try:
        response=requests.post(URL+'/predict',json={'tags':choice},timeout=15);response.raise_for_status();st.json(response.json())
    except Exception as ex:st.error(f'Error de API: {ex}')
st.subheader('Monitoreo de consultas')
try:st.json(requests.get(URL+'/monitor',timeout=10).json())
except Exception:st.warning('No se pudo consultar el monitoreo.')
