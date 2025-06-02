from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from typing import Dict, Union

app = FastAPI(title="Clasificador de Texto - Suicidalidad/Depresión")

# ==== Cargar vectorizador ====
vectorizer = joblib.load('./Modelos_entrenados/vectorizer.joblib')

# ==== Cargar todos los modelos ====
modelos = {}
for file in os.listdir('./Modelos_entrenados'):
    if file.endswith('.joblib') and file != 'vectorizer.joblib':
        name = file.replace('.joblib', '').replace('_', ' ').title()
        modelos[name] = joblib.load(os.path.join('./Modelos_entrenados', file))

# ==== Definir entrada esperada ====
class EntradaTexto(BaseModel):
    title: str
    text: str

@app.get("/")
def root():
    return {"mensaje": "Bienvenido a la API de Clasificación de Texto. Usa /predict para hacer predicciones."}

# ==== Ruta: predicción de todos los modelos ====
@app.post("/predict", response_model=Dict[str, Dict[str, Union[float, int]]])
def predecir_todos(entrada: EntradaTexto):
    texto = entrada.title + " " + entrada.text
    vectorizado = vectorizer.transform([texto])

    resultados = {}
    for nombre, modelo in modelos.items():
        if hasattr(modelo, "predict_proba"):
            prob = modelo.predict_proba(vectorizado)[0][1]
        else:
            prob = float(modelo.predict(vectorizado)[0])

        binario = int(prob >= 0.5)
        resultados[nombre] = {
            "probabilidad": round(prob, 4),
            "prediccion": binario
        }

    return resultados

# ==== Ruta: predicción de un modelo específico ====
@app.post("/predict/{nombre_modelo}", response_model=Dict[str, bool])
def predecir_por_modelo(nombre_modelo: str, entrada: EntradaTexto):
    nombre_formateado = nombre_modelo.replace('_', ' ').title()
    if nombre_formateado not in modelos:
        raise HTTPException(status_code=404, detail=f"Modelo '{nombre_modelo}' no encontrado")

    modelo = modelos[nombre_formateado]
    texto = entrada.title + " " + entrada.text
    vectorizado = vectorizer.transform([texto])

    if hasattr(modelo, "predict_proba"):
        prob = modelo.predict_proba(vectorizado)[0][1]
    else:
        prob = float(modelo.predict(vectorizado)[0])

    prediccion = prob >= 0.5
    return {nombre_modelo: prediccion}
