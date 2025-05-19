#%%
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori as mlxtend_apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
from efficient_apriori import apriori
import seaborn as sns

#%%
# Cargar datos y explorar estructura inicial
data_path = "./data/ItemList.csv"
transacciones = pd.read_csv(data_path)
transacciones

#%% Transformar la lista de transacciones a formato binario
# Crear una lista de listas, donde cada lista contiene los items de una transacción
lista_de_transacciones = []
for index, row in transacciones.iterrows():
    transaccion = [item for item in row[1:].dropna().tolist() if item != 'Missing value']
    lista_de_transacciones.append(transaccion)

print("\nLista de transacciones transformada (primeras 5):")
print(lista_de_transacciones[:5])

#%% Transformar la lista de transacciones a formato binario
te = TransactionEncoder()
te_ary = te.fit(lista_de_transacciones).transform(lista_de_transacciones)
df = pd.DataFrame(te_ary, columns=te.columns_)
df

#%% Se hace el conjunto de frecuencias con un soporte mínimo de 0.01
conjuntos_frecuentes = mlxtend_apriori(df, min_support=0.01, use_colnames=True)
conjuntos_frecuentes

#%% Generar reglas de asociación

# El parámetro metric="confidence" indica que estás interesado en generar reglas basadas en la métrica de confianza.
# El parámetro min_threshold establece un umbral mínimo para la métrica de confianza. Solo se generarán las reglas que tengan una confianza igual o superior a 1%.
reglas = association_rules (conjuntos_frecuentes, metric="confidence", min_threshold=0.01)

# Ordenar las reglas generadas por la métrica de "lift" en orden descendente
reglas_ordenadas = reglas.sort_values("lift", ascending=False)
reglas_ordenadas

#%% Gráfico de Barras - Top Reglas

def generar_grafico_barras_reglas(reglas_df, top_n=10):
    """
    Genera un gráfico de barras horizontal para visualizar las N reglas de asociación
    más importantes, ordenadas por lift.

    Parámetros:
    ----------
    reglas_df : pandas.DataFrame
        DataFrame que contiene las reglas de asociación, con columnas como
        'antecedents', 'consequents', 'lift', etc.
    top_n : int, opcional
        Número de reglas a mostrar en el gráfico. Por defecto, 10.

    Retorna:
    -------
    None (muestra el gráfico)
    """

    # Verificar si el DataFrame está vacío o no contiene las columnas necesarias
    if reglas_df.empty:
        print("Error: El DataFrame de reglas está vacío.")
        return
    
    columnas_requeridas = ['antecedents', 'consequents', 'lift']
    if not all(col in reglas_df.columns for col in columnas_requeridas):
        print(f"Error: El DataFrame debe contener las columnas: {columnas_requeridas}")
        return

    # Crear una nueva columna 'regla_formateada' para las etiquetas del gráfico
    reglas_df["regla_formateada"] = reglas_df.apply(
        lambda x: f"{', '.join(x['antecedents'])} -> {', '.join(x['consequents'])}", axis=1
    )

    # Seleccionar las N reglas principales
    top_reglas = reglas_df.nlargest(top_n, 'lift') #Usamos nlargest

    # Crear el gráfico de barras
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x="lift", y="regla_formateada", data=top_reglas,
                     edgecolor="black", linewidth=1.5, saturation=1)

    plt.title(f"Top {top_n} Reglas de Asociación por Lift", fontsize=16, pad=20)
    plt.xlabel("Lift (Fuerza de asociación)", fontsize=12)
    plt.ylabel("Reglas", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    # Añadir los valores de lift al final de cada barra
    for i, value in enumerate(top_reglas["lift"]):
        ax.text(value + 0.05, i, f"{value:.2f}", va="center", fontsize=10, color="black")

    plt.tight_layout()
    plt.show()

# Llamar a la función para generar el gráfico
generar_grafico_barras_reglas(reglas_ordenadas)

#%%  Gráfico de dispersión - Reglas de Asociación

def generar_grafico_dispersion_reglas(reglas_df):
    """
    Genera un gráfico de dispersión para visualizar la relación entre
    confianza, soporte y lift de las reglas de asociación.

    Parámetros:
    ----------
    reglas_df : pandas.DataFrame
        DataFrame que contiene las reglas de asociación, con columnas
        'antecedents', 'consequents', 'support', 'confidence' y 'lift'.

    Retorna:
    -------
    None (muestra el gráfico)
    """
    # Verificar si el DataFrame está vacío
    if reglas_df.empty:
        print("Error: El DataFrame de reglas está vacío.")
        return

    # Verificar si el DataFrame contiene las columnas necesarias
    columnas_requeridas = ['support', 'confidence', 'lift']
    if not all(col in reglas_df.columns for col in columnas_requeridas):
        print(f"Error: El DataFrame debe contener las columnas: {columnas_requeridas}")
        return
    
    # Crear el gráfico de dispersión
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='confidence', y='support', size='lift', data=reglas_df, alpha=0.7,
                    hue='lift') # Añadimos el parámetro hue
    plt.title('Dispersión de Reglas de Asociación (Confianza vs Soporte, Tamaño por Lift)', fontsize=16, pad=20)
    plt.xlabel('Confianza', fontsize=12)
    plt.ylabel('Soporte', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Llamar a la función para generar los gráficos
generar_grafico_dispersion_reglas(reglas)
# %%
