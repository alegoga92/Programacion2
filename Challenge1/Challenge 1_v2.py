#%%
# Importar librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Cargar datos y explorar estructura inicial
data_path = "./data/breast-cancer-wisconsin.data.csv"
data = pd.read_csv(data_path)
print(data.head())  # Muestra las primeras filas

#%%
# Limpieza de datos
data = data.drop(columns=["id", "Unnamed: 32"])  # Columnas irrelevantes eliminadas
label_encoder = LabelEncoder()
data["diagnosis"] = label_encoder.fit_transform(data["diagnosis"])  # Mapeo de 'M' -> 1 y 'B' -> 0

#%%
# Separar características (X) y etiqueta objetivo (y)
features = data.drop(columns=["diagnosis"])
target = data["diagnosis"]

#%%
# Escalamiento de características
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

#%%
# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, target, test_size=0.2, random_state=42, stratify=target
)

#%%
# Inicio del experimento

# Entrenamiento del modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluación del modelo
predictions = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
class_report = classification_report(y_test, predictions)

# Curva ROC y cálculo del AUC
probabilities = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
auc_score = auc(fpr, tpr)

 # Gráfico de la Curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("Curva ROC")
plt.xlabel("FPR (Tasa de Falsos Positivos)")
plt.ylabel("TPR (Tasa de Verdaderos Positivos)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benigno", "Maligno"], yticklabels=["Benigno", "Maligno"])
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Etiqueta Real")
plt.show()

# Impresión de resultados
print(f"Precisión del modelo: {accuracy:.2f}")
print("Reporte de clasificación:")
print(class_report)