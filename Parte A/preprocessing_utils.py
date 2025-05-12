import pandas as pd
import re
import ftfy
import nltk
import contractions
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

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

def remove_emojis(text):
    """
    Elimina emojis del texto.
    """
    emoji_pattern = re.compile("[" 
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F"
        u"\U0001F780-\U0001F7FF"
        u"\U0001F800-\U0001F8FF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_emojis_df(df):
    """
    Aplica la eliminación de emojis a las columnas 'title' y 'text' del DataFrame.
    """
    df['title'] = df['title'].apply(remove_emojis)
    df['text'] = df['text'].apply(remove_emojis)
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
    Arregla el texto con ftfy y lo convierte a minúsculas en las columnas 'title' y 'text'.
    """
    for col in ['title', 'text']:
        df[col] = df[col].astype(str).apply(ftfy.fix_text).str.lower()
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

def preprocess_text_df(df):
    """
    Ejecuta el pipeline completo de preprocesamiento de texto:
    - Descarga recursos de NLTK.
    - Arregla y convierte a minúsculas.
    - Expande contracciones.
    - Elimina emojis.
    - Limpia puntuación.
    - Elimina stopwords.
    - Lematiza el texto.
    """
    download_nltk_resources()
    df = df[['title', 'text', 'is_suicide']].dropna(subset=['title', 'text'])
    df = fix_and_lowercase_df(df)
    df = expand_contractions_df(df)
    df = remove_emojis_df(df)
    df = clean_text_df(df)
    df = remove_stopwords_df(df)
    df = lemmatize_df(df)
    return df