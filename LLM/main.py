"""
Este script utiliza un modelo LLM (OpenAI GPT) para clasificar textos en señales de suicidalidad o depresión.
- Usa few-shot learning con ejemplos del propio dataset.
- Llama a la API de OpenAI para obtener la predicción para cada registro.
- Calcula métricas de desempeño (accuracy, precision, recall, specificity, F1, AUC).
- Muestra matriz de confusión y curva ROC.
- Exporta resultados y métricas a archivos.
"""

from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# === Configuraciones generales ===
pd.set_option('future.no_silent_downcasting', True)

# === Cargar datos ===
df = pd.read_excel("./Data/data_trainomplete.xlsx")
df = df.dropna(subset=['title', 'text', 'is_suicide'])
df['is_suicide'] = df['is_suicide'].astype(str).str.lower().replace({'no': 0, 'yes': 1})
df['is_suicide'] = df['is_suicide'].astype(int)

# === Cliente OpenAI ===
client = OpenAI(api_key="")  

# === Ejemplos few-shot ===
ejemplos = df.sample(5, random_state=42)

# === Función para construir prompt ===
def construir_prompt(ejemplos, title, text):
    """
    Construye el prompt para el modelo LLM usando ejemplos few-shot y el texto a clasificar.
    """
    instrucciones = (
        "Eres un asistente de salud mental. Dado un título y un texto, clasifica el contenido como:\n"
        "'1' si detectas señales de suicidalidad\n"
        "'0' si solo detectas depresión\n\n"
        "Ejemplos:\n"
    )
    for _, row in ejemplos.iterrows():
        instrucciones += f"Título: {row['title']}\nTexto: {row['text']}\nClasificación: {row['is_suicide']}\n\n"
    instrucciones += f"Título: {title}\nTexto: {text}\nClasificación:"
    return instrucciones

# === Clasificar todos los registros ===
resultados = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    prompt = construir_prompt(ejemplos, row['title'], row['text'])

    try:
        # Llamada a la API de OpenAI para obtener la predicción
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        salida = response.choices[0].message.content.strip()
        pred = int(salida[0]) if salida[0] in ['0', '1'] else -1
    except Exception as e:
        print("Error con OpenAI:", e)
        pred = -1

    resultados.append({
        'title': row['title'],
        'text': row['text'],
        'real': row['is_suicide'],
        'predicted': pred
    })

# === Resultados válidos ===
df_resultado = pd.DataFrame(resultados)
df_valid = df_resultado[df_resultado['predicted'].isin([0, 1])]
y_true = df_valid['real'].astype(int)
y_pred = df_valid['predicted'].astype(int)

# === Métricas ===
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else float('nan')
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
spec = tn / (tn + fp)

print("\n=== MÉTRICAS DEL MODELO ===")
print(f"Accuracy     : {acc:.4f}")
print(f"Precision    : {prec:.4f}")
print(f"Recall       : {rec:.4f}")
print(f"Specificity  : {spec:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"AUC          : {auc:.4f}")

# === Matriz de Confusión ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Depresión (0)', 'Suicidalidad (1)'],
            yticklabels=['Depresión (0)', 'Suicidalidad (1)'])
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.show()

# === Curva ROC ===
if len(set(y_true)) > 1:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    plt.ylabel('Tasa de Verdaderos Positivos (Recall)')
    plt.title('Curva ROC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No se puede trazar la curva ROC porque solo hay una clase presente.")

# === Exportar resultados ===
df_valid.to_csv("predicciones_con_llm.csv", index=False)
print("\n✅ Archivo 'predicciones_con_llm.csv' guardado con éxito.")

# === Guardar métricas en Excel ===
metricas = {
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'AUC'],
    'Valor':   [acc, prec, rec, spec, f1, auc]
}
df_metricas = pd.DataFrame(metricas)
df_metricas.to_excel("metricas_llm.xlsx", index=False)

print("📊 Archivo 'metricas_llm.xlsx' guardado con éxito.")