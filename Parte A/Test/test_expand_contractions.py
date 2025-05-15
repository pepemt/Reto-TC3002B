# test_preprocessing_utils.py
import pandas as pd
import numpy as np
import Utils.preprocessing_utils as pu
import pytest
import re

# 1. Pruebas para funciones de procesamiento de texto

def test_expand_contractions_valid():
    """Caso válido: expande contracciones comunes en el texto."""
    text = "I'm not sure if it won't work"
    expected = "I am not sure if it will not work"  # "I'm" -> "I am", "won't" -> "will not"
    result = pu.expand_contractions(text)
    assert result == expected

def test_expand_contractions_edge_empty():
    """Caso borde: texto vacío debe retornar vacío."""
    assert pu.expand_contractions("") == ""

def test_expand_contractions_invalid_type():
    """Caso inválido: entrada de tipo no cadena debe causar excepción."""
    # Se espera que pasar un tipo incorrecto (int) lance un error (TypeError, AttributeError, etc.)
    with pytest.raises(Exception):
        _ = pu.expand_contractions(123)

def test_replace_emojis_with_text_valid():
    """Caso válido: reemplaza emojis por descripciones de texto."""
    text = "Hello 😊"
    result = pu.replace_emojis_with_text(text)
    # El resultado debe contener la descripción del emoji en formato :emoji_name:
    assert ":smiling_face" in result or ":smile" in result  # acepta cualquiera de los nombres similares
    # Aseguramos que el resto del texto permanece
    assert result.startswith("Hello ")

def test_replace_emojis_with_text_edge_no_emoji():
    """Caso borde: texto sin emojis queda igual."""
    text = "Just a normal text"
    assert pu.replace_emojis_with_text(text) == text

def test_replace_emojis_with_text_invalid_type():
    """Caso inválido: entrada no cadena provoca excepción."""
    with pytest.raises(Exception):
        pu.replace_emojis_with_text(None)

def test_clean_text_valid():
    """Caso válido: elimina caracteres no alfanuméricos."""
    text = "Hello!!! World? (2021)"
    expected = "Hello World 2021"  # Remove punctuation but keep spaces and numbers
    result = pu.clean_text(text)
    assert result == expected

def test_clean_text_edge_only_symbols():
    """Caso borde: texto con solo símbolos debe quedar vacío."""
    text = "$$$!!!"
    expected = ""  # all symbols removed
    result = pu.clean_text(text)
    assert result == expected

def test_clean_text_invalid_type():
    """Caso inválido: tipo incorrecto (None) lanza excepción."""
    with pytest.raises(Exception):
        pu.clean_text(None)

def test_remove_stopwords_valid():
    """Caso válido: elimina stopwords de una frase dada."""
    text = "the cat and the dog"
    stop_words = {"the", "and"}
    expected = "cat dog"
    result = pu.remove_stopwords(text, stop_words)
    assert result == expected

def test_remove_stopwords_edge_all_stopwords():
    """Caso borde: texto compuesto solo de stopwords retorna cadena vacía."""
    text = "and and the the"
    stop_words = {"the", "and"}
    expected = ""  # all words removed
    result = pu.remove_stopwords(text, stop_words)
    assert result == expected

def test_remove_stopwords_invalid_type():
    """Caso inválido: texto no cadena debe causar error."""
    stop_words = {"dummy"}
    with pytest.raises(Exception):
        pu.remove_stopwords(None, stop_words)

def test_get_wordnet_pos_valid_mapping():
    """Caso válido: mapea etiquetas POS de Treebank a WordNet correctamente."""
    # Treebank tags and expected WordNet constants
    assert pu.get_wordnet_pos("JJ") == pu.wordnet.ADJ  # Adjective
    assert pu.get_wordnet_pos("VBD") == pu.wordnet.VERB  # Verb (past tense)
    assert pu.get_wordnet_pos("NN") == pu.wordnet.NOUN  # Noun
    assert pu.get_wordnet_pos("RB") == pu.wordnet.ADV   # Adverb

def test_get_wordnet_pos_edge_unknown():
    """Caso borde: etiqueta no reconocida o vacía retorna NOUN por defecto."""
    assert pu.get_wordnet_pos("XYZ") == pu.wordnet.NOUN
    assert pu.get_wordnet_pos("") == pu.wordnet.NOUN

def test_get_wordnet_pos_invalid_type():
    """Caso inválido: entrada que no es str lanza excepción."""
    with pytest.raises(Exception):
        pu.get_wordnet_pos(None)

def test_lemmatize_text_valid(monkeypatch):
    """Caso válido: lematiza texto usando POS tags simulados."""
    text = "happy cats playing quickly"
    # Simulamos word_tokenize y pos_tag para controlar el resultado
    monkeypatch.setattr(pu, "word_tokenize", lambda txt: txt.split())
    def dummy_pos_tag(tokens):
        # Asignar etiquetas específicas para cada token para probar todas las ramas de get_wordnet_pos
        tag_map = {"happy": "JJ", "cats": "NNS", "playing": "VBG", "quickly": "RB"}
        return [(tok, tag_map.get(tok, "NN")) for tok in tokens]
    monkeypatch.setattr(pu, "pos_tag", dummy_pos_tag)
    # Simulamos un lematizador dummy que realiza lematización básica según el pos:
    class DummyLemmatizer:
        def lemmatize(self, word, pos='n'):
            # pos es 'n', 'v', 'a' o 'r' según get_wordnet_pos
            if pos == 'n':  # noun: quitar 's' final si existe (plural a singular)
                return word[:-1] if word.endswith('s') else word
            elif pos == 'v':  # verb: quitar 'ing' o 'ed' final
                if word.endswith('ing'):
                    return word[:-3]  # remover "ing"
                elif word.endswith('ed'):
                    return word[:-2]  # remover "ed"
                else:
                    return word
            else:
                # para adjetivos/adv u otros, devolver tal cual
                return word
    monkeypatch.setattr(pu, "WordNetLemmatizer", lambda: DummyLemmatizer())
    # Ejecutamos la lematización
    result = pu.lemmatize_text(text, DummyLemmatizer())
    expected = "happy cat play quickly"  # "happy" (adj sin cambio), "cats"->"cat", "playing"->"play", "quickly" (adv igual)
    assert result == expected

def test_lemmatize_text_edge_empty(monkeypatch):
    """Caso borde: texto vacío produce texto vacío."""
    # Patch mínimo para evitar necesitar recursos externos
    monkeypatch.setattr(pu, "word_tokenize", lambda txt: txt.split())
    monkeypatch.setattr(pu, "pos_tag", lambda tokens: [])
    dummy_lemmatizer = type("Dummy", (), {"lemmatize": lambda self, w, pos='n': w})()  # lematizador que devuelve palabra igual
    result = pu.lemmatize_text("", dummy_lemmatizer)
    assert result == ""

def test_lemmatize_text_invalid_type(monkeypatch):
    """Caso inválido: si la entrada no es texto, debe fallar durante tokenización."""
    monkeypatch.setattr(pu, "word_tokenize", lambda txt: txt.split())
    dummy_lemmatizer = type("Dummy", (), {"lemmatize": lambda self, w, pos='n': w})()
    with pytest.raises(Exception):
        pu.lemmatize_text(None, dummy_lemmatizer)

def test_remove_numbers_valid():
    """Caso válido: remueve todos los dígitos de la cadena."""
    text = "abc123def456"
    expected = "abcdef"
    assert pu.remove_numbers(text) == expected

def test_remove_numbers_edge_no_digits():
    """Caso borde: texto sin números permanece igual."""
    text = "abcdef"
    assert pu.remove_numbers(text) == text

def test_remove_numbers_invalid_type():
    """Caso inválido: tipo incorrecto (p.ej. entero) lanza excepción."""
    with pytest.raises(Exception):
        pu.remove_numbers(12345)

def test_remove_written_numbers_valid():
    """Caso válido: elimina números escritos en el texto (ignorando mayúsculas/minúsculas)."""
    text = "One hundred and TWO cats"
    result = pu.remove_written_numbers(text)
    # Debe haber eliminado One, TWO y hundred
    assert "one" not in result.lower()
    assert "two" not in result.lower()
    assert "hundred" not in result.lower()
    # El resto del texto permanece
    assert "and" in result and "cats" in result


def test_remove_written_numbers_edge_no_numbers():
    """Caso borde: texto sin números escritos no cambia."""
    text = "no numeric words here"
    assert pu.remove_written_numbers(text) == text

def test_remove_written_numbers_invalid_type():
    """Caso inválido: entrada no cadena lanza excepción."""
    with pytest.raises(Exception):
        pu.remove_written_numbers(None)


# 2. Pruebas para funciones de DataFrame

def test_expand_contractions_df_valid(monkeypatch):
    """Caso válido: expande contracciones en columnas 'title' y 'text'."""
    df = pd.DataFrame({
        "title": ["I'm fine", "No issues"],
        "text": ["Can't complain", "Everything's OK"],
        "other": [1, 2]  # columna extra para asegurarse que no es afectada
    })
    # Monkeypatch pu.expand_contractions para controlar la expansión sin depender de la librería
    monkeypatch.setattr(pu, "expand_contractions", lambda text: text.replace("I'm", "I am").replace("Can't", "Cannot").replace("can't", "cannot").replace("Everything's", "Everything is"))
    result_df = pu.expand_contractions_df(df.copy())
    # Comprobamos que las contracciones se expandieron correctamente en ambas columnas
    assert result_df.loc[0, "title"] == "I am fine"
    assert result_df.loc[0, "text"] == "Cannot complain" or result_df.loc[0, "text"] == "cannot complain"
    assert result_df.loc[1, "text"] == "Everything is OK"
    # Las demás columnas deben permanecer iguales
    assert "other" in result_df.columns
    assert result_df["other"].tolist() == [1, 2]

def test_expand_contractions_df_edge_empty(monkeypatch):
    """Caso borde: DataFrame con cadenas vacías no produce errores y mantiene vacíos."""
    df = pd.DataFrame({
        "title": ["", "I'm here"],
        "text": ["Can't", ""]
    })
    monkeypatch.setattr(pu, "expand_contractions", lambda text: text.replace("I'm", "I am").replace("Can't", "Cannot").replace("can't", "cannot"))
    result_df = pu.expand_contractions_df(df.copy())
    # Las cadenas vacías permanecen vacías
    assert result_df.loc[0, "title"] == ""
    assert result_df.loc[1, "text"] == ""
    # Otras contracciones se expanden
    assert result_df.loc[1, "title"] == "I am here"
    assert result_df.loc[0, "text"] == "Cannot" or result_df.loc[0, "text"] == "cannot"

def test_expand_contractions_df_invalid_missing_column(monkeypatch):
    """Caso inválido: falta columna 'text' en el DataFrame => KeyError."""
    df = pd.DataFrame({"title": ["I'm fine"]})  # falta 'text'
    monkeypatch.setattr(pu, "expand_contractions", lambda text: text)
    with pytest.raises(KeyError):
        pu.expand_contractions_df(df)

def test_replace_emojis_with_text_df_valid(monkeypatch):
    """Caso válido: reemplaza emojis en columnas 'title' y 'text' del DataFrame."""
    df = pd.DataFrame({
        "title": ["Hi 😃", "No emoji here"],
        "text": ["Check this 😂😂", "Nothing to see"]
    })
    # Monkeypatch emoji.demojize para retorno controlado (reemplaza emoji con :NAME:)
    monkeypatch.setattr(pu.emoji, "demojize", lambda text: text.replace("😃", ":smile:").replace("😂", ":joy:"))
    result_df = pu.replace_emojis_with_text_df(df.copy())
    # Verificar reemplazos
    assert ":smile:" in result_df.loc[0, "title"]
    # Dos emojis 😂😂 se reemplazan por dos descripciones
    assert result_df.loc[0, "text"].count(":joy:") == 2
    # Texto sin emoji permanece igual
    assert result_df.loc[1, "title"] == "No emoji here"
    assert result_df.loc[1, "text"] == "Nothing to see"

def test_replace_emojis_with_text_df_edge_no_emoji(monkeypatch):
    """Caso borde: DataFrame sin emojis permanece igual."""
    df = pd.DataFrame({
        "title": ["Hello", ""],
        "text": ["World", "No emoji"]
    })
    monkeypatch.setattr(pu.emoji, "demojize", lambda text: text)  # no cambia nada
    result_df = pu.replace_emojis_with_text_df(df.copy())
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), df.reset_index(drop=True))

def test_replace_emojis_with_text_df_invalid_missing_column(monkeypatch):
    """Caso inválido: falta columna 'title' en el DataFrame => KeyError."""
    df = pd.DataFrame({"text": ["Hi 😊"]})
    monkeypatch.setattr(pu.emoji, "demojize", lambda text: text)
    with pytest.raises(KeyError):
        pu.replace_emojis_with_text_df(df)

def test_clean_text_df_valid():
    """Caso válido: limpia puntuación en 'title' y 'text'."""
    df = pd.DataFrame({
        "title": ["Hello!!!", "No Punct"],
        "text": ["World??", "Clean"]
    })
    result_df = pu.clean_text_df(df.copy())
    # Verificar que se removió la puntuación
    assert result_df.loc[0, "title"] == "Hello"
    assert result_df.loc[0, "text"] == "World"
    # Texto sin puntuación queda igual
    assert result_df.loc[1, "title"] == "No Punct"
    assert result_df.loc[1, "text"] == "Clean"

def test_clean_text_df_edge_empty():
    """Caso borde: campos vacíos permanecen vacíos."""
    df = pd.DataFrame({
        "title": ["Test", ""],
        "text": ["", "All good"]
    })
    result_df = pu.clean_text_df(df.copy())
    assert result_df.loc[0, "text"] == ""  # permaneció vacío
    assert result_df.loc[1, "title"] == ""  # permaneció vacío

def test_clean_text_df_invalid_missing_column():
    """Caso inválido: falta columna 'text' provoca KeyError."""
    df = pd.DataFrame({"title": ["Hello!"]})
    with pytest.raises(KeyError):
        pu.clean_text_df(df)

def test_fix_and_lowercase_df_valid(monkeypatch):
    """Caso válido: arregla texto con ftfy y convierte a minúsculas."""
    df = pd.DataFrame({
        "title": ["CAFÉ", "Straße"],  # contiene acento y eszett
        "text": ["HELLO World", "123 FOO"]
    })
    # Monkeypatch ftfy.fix_text para que devuelva el texto sin cambios de encoding (simular comportamiento)
    monkeypatch.setattr(pu.ftfy, "fix_text", lambda s: s)
    result_df = pu.fix_and_lowercase_df(df.copy())
    # Debe haber convertido todo a minúsculas
    assert result_df.loc[0, "title"] == "café"
    assert result_df.loc[1, "title"] == "straße"
    assert result_df.loc[0, "text"] == "hello world"
    assert result_df.loc[1, "text"] == "123 foo"

def test_fix_and_lowercase_df_edge_includes_none(monkeypatch):
    """Caso borde: valores None se convierten a 'none' (por astype(str) + lower)."""
    df = pd.DataFrame({
        "title": [None, "OK"],
        "text": ["Hi", None]
    })
    monkeypatch.setattr(pu.ftfy, "fix_text", lambda s: s)
    result_df = pu.fix_and_lowercase_df(df.copy())
    # None -> "None" -> lower -> "none"
    assert result_df.loc[0, "title"] == "none"
    assert result_df.loc[1, "text"] == "none"

def test_fix_and_lowercase_df_invalid_missing_column(monkeypatch):
    """Caso inválido: falta columna 'title' lanza KeyError."""
    df = pd.DataFrame({"text": ["Hello"]})
    monkeypatch.setattr(pu.ftfy, "fix_text", lambda s: s)
    with pytest.raises(KeyError):
        pu.fix_and_lowercase_df(df)

def test_remove_stopwords_df_valid(monkeypatch):
    """Caso válido: elimina stopwords en 'title' y 'text'."""
    df = pd.DataFrame({
        "title": ["the sky is blue", "hello world"],
        "text": ["and the world is round", "no stopwords here"]
    })
    # Monkeypatch stopwords.words para usar una lista conocida en lugar de la real
    monkeypatch.setattr(pu.stopwords, "words", lambda lang: ["the", "and", "is"])
    result_df = pu.remove_stopwords_df(df.copy())
    # Verificar que 'the', 'and', 'is' fueron eliminadas
    assert "the" not in result_df.loc[0, "title"].split()
    assert "and" not in result_df.loc[1, "text"].split()
    # Palabras no stopword permanecen
    assert "sky" in result_df.loc[0, "title"] and "blue" in result_df.loc[0, "title"]
    assert result_df.loc[1, "text"] == "no stopwords here"  # sin cambios porque no contenía stopwords definidas

def test_remove_stopwords_df_edge_all_stopwords(monkeypatch):
    """Caso borde: campos compuestos solo de stopwords quedan vacíos."""
    df = pd.DataFrame({
        "title": ["the and the", "OK"],
        "text": ["and and", "the"]
    })
    monkeypatch.setattr(pu.stopwords, "words", lambda lang: ["the", "and"])
    result_df = pu.remove_stopwords_df(df.copy())
    # Donde todo eran stopwords debe quedar cadena vacía
    assert result_df.loc[0, "title"] == ""
    assert result_df.loc[0, "text"] == ""
    # Campos sin (o con pocas) stopwords se mantienen parcialmente
    assert result_df.loc[1, "title"] == "ok" or result_df.loc[1, "title"] == "OK"  # minúsculas pueden variar si previamente lower
    assert result_df.loc[1, "text"] == ""  # "the" fue removido completamente

def test_remove_stopwords_df_invalid_missing_column(monkeypatch):
    """Caso inválido: falta columna 'text' en DataFrame => KeyError."""
    df = pd.DataFrame({"title": ["hello"]})
    monkeypatch.setattr(pu.stopwords, "words", lambda lang: [])
    with pytest.raises(KeyError):
        pu.remove_stopwords_df(df)

def test_lemmatize_df_valid(monkeypatch):
    """Caso válido: lematiza columnas 'title' y 'text' en el DataFrame."""
    df = pd.DataFrame({
        "title": ["cats and dogs", "running fast"],
        "text": ["playing games", "trees"]
    })
    # Monkeypatch funciones de lematización para control:
    monkeypatch.setattr(pu, "word_tokenize", lambda txt: txt.split())
    # pos_tag asigna 'NNS' a plurales terminados en 's', 'VBG' a terminados en 'ing', 'NN' por defecto.
    def dummy_pos_tag(tokens):
        tagged = []
        for tok in tokens:
            if tok.endswith("s"):
                # si termina en 's' (plural), marcar como NNS (noun plural)
                tagged.append((tok, "NNS"))
            elif tok.endswith("ing"):
                tagged.append((tok, "VBG"))
            else:
                tagged.append((tok, "NN"))
        return tagged
    monkeypatch.setattr(pu, "pos_tag", dummy_pos_tag)
    # Dummy lematizer: singulariza plurales quitando 's', remueve 'ing' de verbos.
    class DummyLemmatizer:
        def lemmatize(self, word, pos='n'):
            if pos == 'n':
                return word[:-1] if word.endswith('s') else word
            elif pos == 'v':
                return word[:-3] if word.endswith('ing') else word
            else:
                return word
    monkeypatch.setattr(pu, "WordNetLemmatizer", lambda: DummyLemmatizer())
    result_df = pu.lemmatize_df(df.copy())
    # Verificar transformaciones:
    # "cats" -> "cat"; "dogs" -> "dog"
    assert result_df.loc[0, "title"] == "cat and dog"
    # "running" -> "run" (dummy removes 'ing'), "fast" sin cambio
    # Nota: según dummy, 'fast' no termina en s/ing, se marcó 'NN' y queda igual
    assert "run" in result_df.loc[1, "title"]
    # "playing" -> "play"
    assert result_df.loc[0, "text"].startswith("play")
    # "trees" -> "tree"
    assert result_df.loc[1, "text"] == "tree"

def test_lemmatize_df_edge_empty(monkeypatch):
    """Caso borde: cadenas vacías siguen vacías tras lematización."""
    df = pd.DataFrame({
        "title": ["", "nothing"],
        "text": ["test", ""]
    })
    monkeypatch.setattr(pu, "word_tokenize", lambda txt: txt.split())
    monkeypatch.setattr(pu, "pos_tag", lambda tokens: [(tok, "NN") for tok in tokens])
    monkeypatch.setattr(pu, "WordNetLemmatizer", lambda: type("Lem", (), {"lemmatize": lambda self, w, pos=None: w})())
    result_df = pu.lemmatize_df(df.copy())
    # Las cadenas vacías permanecen vacías
    assert result_df.loc[0, "title"] == ""
    assert result_df.loc[1, "text"] == ""

def test_lemmatize_df_invalid_missing_column(monkeypatch):
    """Caso inválido: falta columna 'title' provoca KeyError."""
    df = pd.DataFrame({"text": ["words"]})
    monkeypatch.setattr(pu, "WordNetLemmatizer", lambda: type("Lem", (), {"lemmatize": lambda self, w, pos=None: w})())
    with pytest.raises(KeyError):
        pu.lemmatize_df(df)

def test_eliminate_empty_rows_valid():
    """Caso válido: elimina filas con 'title', 'text' o 'is_suicide' nulos."""
    df = pd.DataFrame({
        "title": ["A", None, "C"],
        "text": ["foo", "bar", None],
        "is_suicide": [1, 0, 1],
        "extra": [100, 200, 300]
    })
    result_df = pu.eliminate_empty_rows(df.copy())
    # Filas con índices 0 y 1 tienen None en al menos una columna (row1 title None, row2 text None),
    # Bueno, en este caso: row0: (A, foo, 1) ok; row1: title None -> drop; row2: text None -> drop.
    # Solo debe quedar fila index 0 (original).
    assert len(result_df) == 1
    # Debe conservar la fila completa de índice 0 original
    assert result_df.iloc[0]["title"] == "A"
    assert result_df.iloc[0]["text"] == "foo"
    assert result_df.iloc[0]["is_suicide"] == 1
    # La columna extra también debe conservarse para esa fila
    assert "extra" in result_df.columns and result_df.iloc[0]["extra"] == 100

def test_eliminate_empty_rows_edge_all_empty():
    """Caso borde: si todas las filas tienen algún campo vacío, resultado es DataFrame vacío."""
    df = pd.DataFrame({
        "title": [None, ""],
        "text": ["", None],
        "is_suicide": [np.nan, np.nan]
    })
    result_df = pu.eliminate_empty_rows(df.copy())
    # Ambas filas tienen al menos un campo vacío (en la fila0 'title' None, fila1 'text' None),
    # deben eliminarse, resultando en 0 filas.
    assert result_df.empty
    # Las columnas deben seguir estando presentes
    for col in ["title", "text", "is_suicide"]:
        assert col in result_df.columns

def test_eliminate_empty_rows_invalid_missing_column():
    """Caso inválido: falta columna 'is_suicide' provoca KeyError."""
    df = pd.DataFrame({
        "title": ["Hi"],
        "text": ["Hello"]
        # 'is_suicide' missing
    })
    with pytest.raises(KeyError):
        pu.eliminate_empty_rows(df)

def test_remove_numbers_df_valid():
    """Caso válido: remueve dígitos en columnas 'title' y 'text'."""
    df = pd.DataFrame({
        "title": ["abc123", "no digits"],
        "text": ["456test", "7 8 9"]
    })
    result_df = pu.remove_numbers_df(df.copy())
    
    assert result_df.loc[0, "title"] == "abc"
    assert result_df.loc[0, "text"] == "test"
    # Validamos que no hay números, sin importar los espacios
    assert re.search(r'\d', result_df.loc[1, "text"]) is None
    assert result_df.loc[1, "title"] == "no digits"

def test_remove_numbers_df_edge_no_digits():
    """Caso borde: sin dígitos, DataFrame queda igual."""
    df = pd.DataFrame({
        "title": ["abc", ""],
        "text": ["hello", "world"]
    })
    result_df = pu.remove_numbers_df(df.copy())
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), df.reset_index(drop=True))

def test_remove_numbers_df_invalid_missing_column():
    """Caso inválido: falta columna 'text' lanza KeyError."""
    df = pd.DataFrame({"title": ["123"]})
    with pytest.raises(KeyError):
        pu.remove_numbers_df(df)

def test_remove_written_numbers_df_valid():
    """Caso válido: elimina números escritos ('one','two', etc.) en el DataFrame."""
    df = pd.DataFrame({
        "title": ["one dog and two cats", "nothing"],
        "text": ["three birds", "five"]
    })
    result_df = pu.remove_written_numbers_df(df.copy())
    # Comprobar que números escritos fueron eliminados (ignorando mayúsculas):
    for word in ["one", "two", "three", "five"]:
        # Debe haberse eliminado independiente de mayúsculas (aquí están en minúscula ya)
        assert word not in result_df.loc[0, "title"].lower()
        assert word not in result_df.loc[1, "text"].lower()
    # Palabras no numéricas siguen presentes
    assert "dog" in result_df.loc[0, "title"] and "cats" in result_df.loc[0, "title"]
    assert result_df.loc[1, "title"] == "nothing"

def test_remove_written_numbers_df_edge_no_numbers():
    """Caso borde: sin palabras numéricas, DataFrame permanece igual."""
    df = pd.DataFrame({
        "title": ["hello world"],
        "text": [""]
    })
    result_df = pu.remove_written_numbers_df(df.copy())
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), df.reset_index(drop=True))

def test_remove_written_numbers_df_invalid_missing_column():
    """Caso inválido: falta columna 'title' lanza KeyError."""
    df = pd.DataFrame({"text": ["one"]})
    with pytest.raises(KeyError):
        pu.remove_written_numbers_df(df)

def test_preprocess_text_df_pipeline(monkeypatch):
    """Caso válido: prueba integrada del pipeline completo de preprocess_text_df."""
    # DataFrame de entrada con diversas necesidades de limpieza
    df = pd.DataFrame({
        "title": ["I'm 😊 123 running to one", None],
        "text": ["Testing... 456 and TWO", "All good"],
        "is_suicide": [0, 1],
        "extra": ["keep", "keep"]  # columna extra que debería conservarse tras seleccionar ['title','text','is_suicide']
    })
    # Definimos funciones dummy para cada paso del pipeline, para controlar la transformación:
    def dummy_download_nltk():
        calls.append("download")
    def dummy_eliminate_empty_rows(df_in):
        calls.append("eliminate_empty")
        # Eliminar filas con NA en title/text/is_suicide:
        return df_in.dropna(subset=["title", "text", "is_suicide"])
    def dummy_fix_and_lowercase_df(df_in):
        calls.append("fix_lower")
        df_out = df_in.copy()
        df_out["title"] = df_out["title"].astype(str).str.lower()
        df_out["text"] = df_out["text"].astype(str).str.lower()
        return df_out
    def dummy_expand_contractions_df(df_in):
        calls.append("expand")
        df_out = df_in.copy()
        # Expandir contracciones básicas
        df_out["title"] = df_out["title"].str.replace("I'm", "I am", regex=False)
        df_out["text"] = df_out["text"].str.replace("I'm", "I am", regex=False)
        df_out["title"] = df_out["title"].str.replace("can't", "cannot", regex=False)
        df_out["text"] = df_out["text"].str.replace("can't", "cannot", regex=False)
        return df_out
    def dummy_replace_emojis_with_text_df(df_in):
        calls.append("replace_emojis")
        df_out = df_in.copy()
        # Reemplazar emoji 😊 por texto ':smiling_face:' (agregado para luego quitar puntuación)
        df_out["title"] = df_out["title"].str.replace("😊", ":smiling_face:", regex=False)
        df_out["text"] = df_out["text"].str.replace("😊", ":smiling_face:", regex=False)
        return df_out
    def dummy_clean_text_df(df_in):
        calls.append("clean")
        df_out = df_in.copy()
        # Remover puntuación (incluye los ':' introducidos por emojis)
        df_out["title"] = df_out["title"].str.replace(r"[^\w\s]", "", regex=True)
        df_out["text"] = df_out["text"].str.replace(r"[^\w\s]", "", regex=True)
        return df_out
    def dummy_remove_numbers_df(df_in):
        calls.append("remove_nums")
        df_out = df_in.copy()
        df_out["title"] = df_out["title"].str.replace(r"\d+", "", regex=True)
        df_out["text"] = df_out["text"].str.replace(r"\d+", "", regex=True)
        return df_out
    def dummy_remove_written_numbers_df(df_in):
        calls.append("remove_written_nums")
        df_out = df_in.copy()
        # Remover palabras 'one','two','two' etc (ignorando mayúsculas)
        pattern = r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|two)\b"
        df_out["title"] = df_out["title"].str.replace(pattern, "", regex=True, case=False)
        df_out["text"] = df_out["text"].str.replace(pattern, "", regex=True, case=False)
        return df_out
    def dummy_remove_stopwords_df(df_in):
        calls.append("remove_stopwords")
        stop_set = {"and", "to", "am"}  # conjunto de stopwords simulado
        df_out = df_in.copy()
        df_out["title"] = df_out["title"].apply(lambda txt: " ".join([w for w in str(txt).split() if w not in stop_set]))
        df_out["text"] = df_out["text"].apply(lambda txt: " ".join([w for w in str(txt).split() if w not in stop_set]))
        return df_out
    def dummy_lemmatize_df(df_in):
        calls.append("lemmatize")
        df_out = df_in.copy()
        # Lematizador dummy: quitar 's' finales y 'ing' finales de cada palabra
        def lemm_sentence(sentence):
            words = str(sentence).split()
            lemmed = []
            for w in words:
                if w.endswith("s"):
                    lemmed.append(w[:-1])  # plural -> singular
                elif w.endswith("ing"):
                    lemmed.append(w[:-3])  # remove 'ing'
                else:
                    lemmed.append(w)
            return " ".join(lemmed)
        df_out["title"] = df_out["title"].apply(lemm_sentence)
        df_out["text"] = df_out["text"].apply(lemm_sentence)
        return df_out

    # Lista para rastrear llamadas y su orden
    calls = []
    # Monkeypatch cada sub-función llamada dentro de preprocess_text_df
    monkeypatch.setattr(pu, "download_nltk_resources", dummy_download_nltk)
    monkeypatch.setattr(pu, "eliminate_empty_rows", dummy_eliminate_empty_rows)
    monkeypatch.setattr(pu, "fix_and_lowercase_df", dummy_fix_and_lowercase_df)
    monkeypatch.setattr(pu, "expand_contractions_df", dummy_expand_contractions_df)
    monkeypatch.setattr(pu, "replace_emojis_with_text_df", dummy_replace_emojis_with_text_df)
    monkeypatch.setattr(pu, "clean_text_df", dummy_clean_text_df)
    monkeypatch.setattr(pu, "remove_numbers_df", dummy_remove_numbers_df)
    monkeypatch.setattr(pu, "remove_written_numbers_df", dummy_remove_written_numbers_df)
    monkeypatch.setattr(pu, "remove_stopwords_df", dummy_remove_stopwords_df)
    monkeypatch.setattr(pu, "lemmatize_df", dummy_lemmatize_df)

    # Ejecutar el pipeline completo
    result_df = pu.preprocess_text_df(df.copy())
    # Verificar que se llamaron todos los pasos en el orden esperado
    expected_call_order = ["download", "eliminate_empty", "fix_lower", "expand", "replace_emojis",
                           "clean", "remove_nums", "remove_written_nums", "remove_stopwords", "lemmatize"]
    assert calls == expected_call_order

    # Construir DataFrame esperado aplicando manualmente las transformaciones dummy en secuencia:
    df_expected = df.copy()
    df_expected = df_expected.dropna(subset=["title", "text", "is_suicide"])
    df_expected = df_expected[["title", "text", "is_suicide"]]  # descartar columna 'extra'
    df_expected = dummy_fix_and_lowercase_df(df_expected)
    df_expected = dummy_expand_contractions_df(df_expected)
    df_expected = dummy_replace_emojis_with_text_df(df_expected)
    df_expected = dummy_clean_text_df(df_expected)
    df_expected = dummy_remove_numbers_df(df_expected)
    df_expected = dummy_remove_written_numbers_df(df_expected)
    df_expected = dummy_remove_stopwords_df(df_expected)
    df_expected = dummy_lemmatize_df(df_expected)

    # Resetear índices para comparación justa (ya que eliminamos filas)
    result_df = result_df.reset_index(drop=True)
    df_expected = df_expected.reset_index(drop=True)
    # Comparar DataFrames final resultante vs esperado
    pd.testing.assert_frame_equal(result_df, df_expected)
    # Asegurar que la columna 'extra' fue eliminada en el proceso (solo quedan title, text, is_suicide)
    assert list(result_df.columns) == ["title", "text", "is_suicide"]
