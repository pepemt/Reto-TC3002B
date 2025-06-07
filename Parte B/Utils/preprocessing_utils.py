"""
Funciones de preprocesamiento de texto para clasificación:
Incluye expansión de contracciones, reemplazo de emojis, limpieza de texto,
eliminación de números (dígitos y escritos), stopwords, lematización y utilidades para DataFrames.
"""

import pandas as pd
import re
import ftfy
import nltk # type: ignore
import unicodedata 
import contractions # type: ignore
import emoji
from nltk.tokenize import word_tokenize # type: ignore
from nltk import pos_tag 
from nltk.corpus import stopwords, wordnet # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore

def download_nltk_resources():
    """
    Descarga los recursos necesarios de NLTK.
    """
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')


def expand_contractions(text):
    """
    Expande las contracciones en el texto.
    """
    return contractions.fix(text)

def expand_contractions_df(df):
    """
    Aplica la expansión de contracciones a las columnas 'title' y 'text' del DataFrame.
    """
    df['title'] = df['title'].apply(expand_contractions)
    df['text'] = df['text'].apply(expand_contractions)
    return df

def replace_emojis_with_text(text):
    """
    Reemplaza los emojis en el texto con palabras que los describan.
    """
    return emoji.demojize(text)

def replace_emojis_with_text_df(df):
    """
    Aplica el reemplazo de emojis por palabras descriptivas a las columnas 'title' y 'text' del DataFrame.
    """
    df['title'] = df['title'].apply(replace_emojis_with_text)
    df['text'] = df['text'].apply(replace_emojis_with_text)
    return df

def clean_text(text):
    """
    Elimina caracteres no alfanuméricos del texto.
    """
    return re.sub(r'[^\w\s]', '', text)

def clean_text_df(df):
    """
    Aplica la limpieza de texto a las columnas 'title' y 'text' del DataFrame.
    """
    df['title'] = df['title'].apply(clean_text)
    df['text'] = df['text'].apply(clean_text)
    return df

def fix_and_lowercase_df(df):
    """
    Arregla errores de codificación usando ftfy en las columnas 'title' y 'text'.
    No convierte a minúsculas porque el tokenizador de BERT uncased ya lo hace.
    """
    for col in ['title', 'text']:
        df[col] = df[col].astype(str).apply(ftfy.fix_text)
    return df


def remove_stopwords(text, stop_words):
    """
    Elimina las stopwords del texto.
    """
    return ' '.join([word for word in text.split() if word not in stop_words])

def remove_stopwords_df(df):
    """
    Aplica la eliminación de stopwords a las columnas 'title' y 'text' del DataFrame.
    """
    stop_words = set(stopwords.words('english'))
    df['title'] = df['title'].apply(lambda text: remove_stopwords(text, stop_words))
    df['text'] = df['text'].apply(lambda text: remove_stopwords(text, stop_words))
    return df

def get_wordnet_pos(treebank_tag):
    """
    Mapea las etiquetas POS de Treebank a las etiquetas de WordNet.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_text(text, lemmatizer):
    """
    Lematiza el texto usando las etiquetas POS.
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged]
    return ' '.join(lemmatized)

def lemmatize_df(df):
    """
    Aplica la lematización a las columnas 'title' y 'text' del DataFrame.
    """
    lemmatizer = WordNetLemmatizer()
    df['title'] = df['title'].apply(lambda text: lemmatize_text(text, lemmatizer))
    df['text'] = df['text'].apply(lambda text: lemmatize_text(text, lemmatizer))
    return df

def eliminate_empty_rows(df):
    """
    Elimina filas vacías del DataFrame.
    """
    df = df.dropna(subset=['title', 'text', 'is_suicide'])
    return df

def remove_numbers(text):
    """
    Elimina los números del texto y limpia espacios redundantes.
    """
    text = re.sub(r'\d+', '', text)
    return ' '.join(text.split())  # Eliminar espacios redundantes

def remove_numbers_df(df):
    """
    Aplica la eliminación de números a las columnas 'title' y 'text' del DataFrame.
    """
    df['title'] = df['title'].apply(remove_numbers)
    df['text'] = df['text'].apply(remove_numbers)
    return df

def remove_written_numbers(text):
    """
    Elimina números escritos a mano (como 'one', 'two', 'eleven') del texto.
    """
    written_numbers = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", 
        "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", 
        "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion"
    ]
    # Crear un patrón para buscar palabras numéricas
    pattern = r'\b(?:' + '|'.join(written_numbers) + r')\b'
    # Reemplazar las palabras numéricas con una cadena vacía
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    # Eliminar espacios redundantes
    return ' '.join(text.split())

def remove_written_numbers_df(df):
    """
    Aplica la eliminación de números escritos a mano a las columnas 'title' y 'text' del DataFrame.
    """
    df['title'] = df['title'].apply(remove_written_numbers)
    df['text'] = df['text'].apply(remove_written_numbers)
    return df

def preprocess_text_df(df):
    """
    Ejecuta el pipeline completo de preprocesamiento de texto:
    - Descarga recursos de NLTK.
    - Arregla y convierte a minúsculas.
    - Expande contracciones.
    - Reemplaza emojis por palabras descriptivas.
    - Limpia puntuación.
    - Elimina números y números escritos a mano.
    - Elimina stopwords.
    - Lematiza el texto.
    """
    #download_nltk_resources()
    df = eliminate_empty_rows(df)
    df = df[['title', 'text', 'is_suicide']].dropna(subset=['title', 'text'])
    df = fix_and_lowercase_df(df)
    df = expand_contractions_df(df)
    df = replace_emojis_with_text_df(df)
    df = clean_text_df(df)
    df = remove_numbers_df(df)  # Elimina números
    df = remove_written_numbers_df(df)  # Elimina números escritos a mano
    df = remove_stopwords_df(df)
    df = lemmatize_df(df)
    return df


def preprocess_text_df_bert(df):
    """
    Preprocesamiento mínimo recomendado para BERT/MentalBERT.
    """
    df = eliminate_empty_rows(df)
    df = df[['title', 'text', 'is_suicide']].dropna(subset=['title', 'text'])
    df = fix_and_lowercase_df(df)  # Solo para limpiar encoding con ftfy
    df = expand_contractions_df(df)
    df = replace_emojis_with_text_df(df)
    return df
