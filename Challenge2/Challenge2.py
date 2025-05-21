import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
from langdetect import detect, DetectorFactory
from pysentimiento import create_analyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import numpy as np

# --- Parte 1: Web Scraping (código proporcionado por el usuario) ---

url = 'https://es.restaurantguru.com/Review-Zapopan/reviews?bylang=1'
reviewlist = []

def get_soup(url):
    """
    Realiza una solicitud HTTP y devuelve un objeto BeautifulSoup.
    Asume que un servidor Splash/Scrapy-Splash está corriendo en localhost:8050.
    """
    try:
        page = requests.get('http://localhost:8050/render.html', params={'url': url, 'wait': 2})
        if page.status_code == 200:
            soup = BeautifulSoup(page.text, "html.parser")
            return soup
        else:
            print(f"Error: {page.status_code}")
            return None
    except Exception as e:
        print(f"Error al conectar: {e}")
        return None

def get_reviews(soup):
    """
    Extrae reseñas de un objeto BeautifulSoup y las añade a reviewlist.
    """
    if soup:
        container = soup.find('div', {'class': 'scroll-container clear wrapper_reviews'})
        if container:
            reviews = container.find_all('div', {'class': 'o_review'})
            for item in reviews:
                review = {
                    'comment': item.find('span', {'class': 'text_full'}).text.strip() if item.find('span', {'class': 'text_full'}) else 'Sin comentario'
                }
                reviewlist.append(review)
        else:
            print("No se encontró el contenedor con id='comments_conteiner'.")

soup = get_soup(url)
if soup:
     get_reviews(soup)

print(f"Cantidad de reseñas encontradas: {len(reviewlist)}")

for review in reviewlist:
     print(review)

df = pd.DataFrame(reviewlist)
print(df.head())

df = pd.DataFrame(data)
print("DataFrame inicial (ejemplo):")
print(df)
print("-" * 50)

# --- Parte 2: Stage 2 - Clasificador y Análisis de Sentimientos ---

# 1. Crear DataFrame Inglés-Español y analizar por separado

# Detectar el idioma de cada comentario
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'

df['language'] = df['comment'].apply(detect_language)

# Separar DataFrames por idioma
df_spanish = df[df['language'] == 'es'].copy()
df_english = df[df['language'] == 'en'].copy()

print(f"Reseñas en Español: {len(df_spanish)}")
print(f"Reseñas en Inglés: {len(df_english)}")
print("-" * 50)

# 2. Limpieza de datos
def clean_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'[^a-záéíóúüñ\s]', '', text) # Eliminar puntuación y números (para español)
    text = re.sub(r'[^a-z\s]', '', text) # Eliminar puntuación y números (para inglés)
    text = re.sub(r'\s+', ' ', text).strip()  # Eliminar espacios extra
    return text

df_spanish['cleaned_comment'] = df_spanish['comment'].apply(clean_text)
df_english['cleaned_comment'] = df_english['comment'].apply(clean_text)

print("Comentarios limpios (ejemplo español):")
print(df_spanish[['comment', 'cleaned_comment']].head())
print("-" * 50)

# 3. Stop words y Lemmatization

# Descargar recursos de NLTK si no están disponibles
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except nltk.downloader.DownloadError:
    nltk.download('omw-1.4')

# Cargar stop words
stop_words_spanish = set(stopwords.words('spanish'))
stop_words_english = set(stopwords.words('english'))

# Inicializar lematizador
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, lang='es'):
    tokens = text.split()
    if lang == 'es':
        tokens = [word for word in tokens if word not in stop_words_spanish]
        # NLTK WordNetLemmatizer no es ideal para español, pero lo usamos para consistencia
        # Una alternativa mejor sería spaCy con un modelo español.
        # Por simplicidad, y dado que pysentimiento ya maneja bien el español,
        # nos enfocaremos en la limpieza y stop words para el clasificador.
        # Si se usa spaCy:
        # import spacy
        # nlp_es = spacy.load("es_core_news_sm")
        # doc = nlp_es(text)
        # return " ".join([token.lemma_ for token in doc if token.text not in stop_words_spanish])
    elif lang == 'en':
        tokens = [word for word in tokens if word not in stop_words_english]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

df_spanish['processed_comment'] = df_spanish['cleaned_comment'].apply(lambda x: preprocess_text(x, 'es'))
df_english['processed_comment'] = df_english['cleaned_comment'].apply(lambda x: preprocess_text(x, 'en'))

print("Comentarios preprocesados (ejemplo español):")
print(df_spanish[['comment', 'processed_comment']].head())
print("-" * 50)

# 4. Distribuciones de N-gramas
def get_ngrams(text_series, n=2, top_k=10):
    all_ngrams = []
    for text in text_series:
        tokens = text.split()
        all_ngrams.extend(list(ngrams(tokens, n)))
    return Counter(all_ngrams).most_common(top_k)

print("Top 5 Bigramas en Español:")
print(get_ngrams(df_spanish['processed_comment'], n=2, top_k=5))
print("\nTop 5 Bigramas en Inglés:")
print(get_ngrams(df_english['processed_comment'], n=2, top_k=5))
print("-" * 50)

# 5. Análisis de Sentimientos (pysentimiento)

# Inicializar el analizador de sentimientos para español
analyzer = create_analyzer(task="sentiment", lang="es")

# Aplicar análisis de sentimientos a las reseñas en español
def get_sentiment_pysentimiento(text):
    if pd.isna(text) or text.strip() == '':
        return {'output': 'NEU', 'probas': {'NEU': 1.0, 'POS': 0.0, 'NEG': 0.0}}
    result = analyzer.predict(text)
    return {'output': result.output, 'probas': result.probas}

df_spanish['sentiment_pysentimiento'] = df_spanish['comment'].apply(get_sentiment_pysentimiento)
df_spanish['sentiment_label'] = df_spanish['sentiment_pysentimiento'].apply(lambda x: x['output'])
df_spanish['sentiment_score_pos'] = df_spanish['sentiment_pysentimiento'].apply(lambda x: x['probas']['POS'])
df_spanish['sentiment_score_neg'] = df_spanish['sentiment_pysentimiento'].apply(lambda x: x['probas']['NEG'])
df_spanish['sentiment_score_neu'] = df_spanish['sentiment_pysentimiento'].apply(lambda x: x['probas']['NEU'])

print("Análisis de Sentimientos (Español) con pysentimiento:")
print(df_spanish[['comment', 'sentiment_label', 'sentiment_score_pos', 'sentiment_score_neg']].head())
print("-" * 50)

# Para el propósito de la clasificación, necesitamos una etiqueta numérica.
# Mapeamos las etiquetas de sentimiento a números: POS=1, NEU=0, NEG=-1
# O para clasificación binaria: POS=1, NEG/NEU=0
# Para este ejemplo, crearemos una etiqueta binaria simplificada para la clasificación.
# Consideraremos 'POS' como positivo (1) y 'NEG'/'NEU' como no positivo (0).
df_spanish['target'] = df_spanish['sentiment_label'].apply(lambda x: 1 if x == 'POS' else 0)

# --- Clasificación propuesta ---
# Construcción del modelo, entrenamiento y cálculo de probabilidades gramaticales

# Asegurarse de tener suficientes datos para dividir
if len(df_spanish) < 2:
    print("Advertencia: No hay suficientes datos en español para entrenar un modelo de clasificación.")
    print("Se necesita al menos 2 muestras para dividir en conjuntos de entrenamiento y prueba.")
    # Si no hay suficientes datos, salimos de la parte de clasificación
    X_train, X_test, y_train, y_test = [], [], [], []
else:
    X = df_spanish['processed_comment']
    y = df_spanish['target']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)

    # Vectorización de texto (Extracción de características principales)
    # TF-IDF para convertir texto en características numéricas
    vectorizer = TfidfVectorizer(max_features=1000) # Limitar a 1000 características
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Construcción y entrenamiento del modelo (Clasificador Naive Bayes Multinomial)
    # Este modelo es adecuado para datos de conteo de texto (como TF-IDF)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Predicciones
    y_pred = model.predict(X_test_vec)

    # Evaluación del modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("\nResultados del Modelo de Clasificación (Español):")
    print(f"Precisión: {accuracy:.2f}")
    print("\nReporte de Clasificación:\n", report)
    print("\nMatriz de Confusión:\n", conf_matrix)
    print("-" * 50)

    # Cálculo de probabilidades gramaticales (implícito en TF-IDF y el modelo)
    # TF-IDF ya captura la importancia de las palabras (probabilidades relativas)
    # para la clasificación. El modelo Naive Bayes usa estas probabilidades para clasificar.

# --- Parte 3: Stage 3 - Pipeline de MLOps con MLflow ---

# Configurar el tracking URI de MLflow
# Asegúrate de que el servidor MLflow esté corriendo (mlflow ui)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Restaurant_Review_Sentiment_Analysis")

# Iniciar una nueva ejecución de MLflow
with mlflow.start_run(run_name="Sentiment_Classifier_Run"):
    # Loguear parámetros del modelo (si el modelo fue entrenado)
    if 'model' in locals():
        mlflow.log_param("model_name", "Multinomial Naive Bayes")
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("max_features", 1000)
        mlflow.log_param("test_size", 0.2)

        # Loguear métricas
        mlflow.log_metric("accuracy", accuracy)
        # Puedes loguear otras métricas del classification_report si las parseas
        # Por ejemplo, F1-score para la clase positiva
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        mlflow.log_metric("f1_score_pos", report_dict['1']['f1-score'])
        mlflow.log_metric("precision_pos", report_dict['1']['precision'])
        mlflow.log_metric("recall_pos", report_dict['1']['recall'])

        # Guardar el plot de la matriz de confusión como un artefacto
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Positivo', 'Positivo'], yticklabels=['No Positivo', 'Positivo'])
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.title('Matriz de Confusión')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close() # Cierra la figura para liberar memoria

        # Guardar un plot de distribución de N-gramas como artefacto
        top_bigrams_es = get_ngrams(df_spanish['processed_comment'], n=2, top_k=10)
        if top_bigrams_es:
            bigram_labels = [" ".join(ngram[0]) for ngram in top_bigrams_es]
            bigram_counts = [ngram[1] for ngram in top_bigrams_es]

            plt.figure(figsize=(10, 6))
            sns.barplot(x=bigram_counts, y=bigram_labels, palette='viridis')
            plt.title('Top 10 Bigramas en Reseñas en Español')
            plt.xlabel('Frecuencia')
            plt.ylabel('Bigrama')
            plt.tight_layout()
            plt.savefig("top_bigrams_spanish.png")
            mlflow.log_artifact("top_bigrams_spanish.png")
            plt.close()

        # Loguear el modelo con su firma
        # La firma del modelo describe las entradas y salidas esperadas del modelo
        # Para un modelo scikit-learn, mlflow.sklearn.log_model lo infiere automáticamente
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sentiment_model",
            registered_model_name="RestaurantSentimentClassifier" # Opcional: registrar el modelo en el Model Registry
        )

        print("\nMLflow Run completado. Visita http://localhost:5000 para ver los resultados.")
    else:
        print("\nEl modelo de clasificación no fue entrenado debido a la falta de datos.")
        print("MLflow Run no se inició para la parte de clasificación.")