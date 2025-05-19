#%%
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import joblib
from mlflow.tracking import MlflowClient

#%%
# Configuración de paths - ACTUALIZADA PARA USAR LA RUTA ESPECIFICADA
BASE_DIR = r"C:/Users/alego/OneDrive/Documentos/MCD/Semestre2/Programacion2/Programacion2/Challenge1"
DATA_PATH = os.path.join(BASE_DIR, "data", "breast-cancer-wisconsin.data.csv")
MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")  # Esta es la ruta exacta que solicitaste
SCALER_PATH = os.path.join(BASE_DIR, "scripts", "scaler.save")

def setup_mlflow():
    """Configurar MLflow tracking con la ruta exacta especificada"""
    # Asegurar que el directorio mlruns existe
    os.makedirs(MLFLOW_TRACKING_URI, exist_ok=True)
    
    # Configurar MLflow con la URI de file://
    mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_URI}")
    mlflow.set_experiment("Breast_Cancer_Wisconsin")

def load_data(file_path):
    """Cargar y preparar los datos"""
    df = pd.read_csv(file_path)
    df = df.drop(columns=["id", "Unnamed: 32"])
    df["diagnosis"] = LabelEncoder().fit_transform(df["diagnosis"])
    return df

def preprocess_data(df, save_scaler=False):
    """Preprocesamiento de datos"""
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if save_scaler:
        # Asegurar que el directorio scripts existe
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y), scaler

def train_model(X_train, y_train):
    """Entrenar el modelo"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, run_id=None):
    """Evaluar el modelo y registrar métricas"""
    active_run = None
    
    try:
        if run_id:
            active_run = mlflow.start_run(run_id=run_id)
        else:
            active_run = mlflow.start_run()
            run_id = active_run.info.run_id
        
        # Hacer predicciones
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Registrar métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_text(report, "classification_report.txt")
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Guardar y registrar gráficos
        plot_roc_curve(fpr, tpr, roc_auc)
        plot_confusion_matrix(y_test, y_pred)
        
        print(f"Evaluación completada - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
        
        return accuracy, roc_auc, run_id
    
    finally:
        if active_run:
            mlflow.end_run()

def plot_roc_curve(fpr, tpr, roc_auc):
    """Graficar y guardar curva ROC"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig("roc_curve.png")
    plt.close()
    mlflow.log_artifact("roc_curve.png")

def plot_confusion_matrix(y_test, y_pred):
    """Graficar y guardar matriz de confusión"""
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benigno', 'Maligno'], 
                yticklabels=['Benigno', 'Maligno'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

def get_latest_model():
    """Obtener el último modelo entrenado desde MLflow"""
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Breast_Cancer_Wisconsin")
    
    if experiment is None:
        raise ValueError("No se encontró el experimento en MLflow")
    
    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
    
    if not runs:
        raise ValueError("No se encontraron modelos entrenados en MLflow")
    
    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri), run_id

def predict_new_data(model, input_data):
    """Realizar predicciones con nuevos datos"""
    # Preprocesar datos
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(input_data)
    
    # Hacer predicción
    prediction = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    return prediction, probabilities

def log_prediction(run_id, prediction, probabilities):
    """Registrar predicción en MLflow"""
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("latest_prediction", int(prediction[0]))
        mlflow.log_metric("probability_benign", probabilities[0][0])
        mlflow.log_metric("probability_malignant", probabilities[0][1])

def run_pipeline():
    """Ejecutar el pipeline completo: entrenamiento, evaluación y predicción"""
    # Configurar MLflow con la ruta exacta especificada
    setup_mlflow()
    
    # Paso 1: Entrenamiento del modelo
    with mlflow.start_run(run_name="Training_Run") as run:
        # Cargar y preparar datos
        df = load_data(DATA_PATH)
        (X_train, X_test, y_train, y_test), scaler = preprocess_data(df, save_scaler=True)
        
        # Registrar parámetros
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        
        # Guardar el scaler como artefacto
        mlflow.log_artifact(SCALER_PATH, "scaler")
        
        # Entrenar modelo
        model = train_model(X_train, y_train)
        
        # Registrar modelo
        mlflow.sklearn.log_model(model, "model")
        
        run_id = run.info.run_id
        print(f"Modelo entrenado y registrado en MLflow. Run ID: {run_id}")
        print(f"Los resultados se han guardado en: {MLFLOW_TRACKING_URI}")
    
    # Paso 2: Evaluación del modelo
    print("\nEvaluando modelo...")
    evaluate_model(model, X_test, y_test, run_id)
    
    # Paso 3: Predicción con nuevos datos
    print("\nRealizando predicción con datos de ejemplo...")
    example_data = {
        'radius_mean': [17.99], 'texture_mean': [10.38], 'perimeter_mean': [122.8],
        'area_mean': [1001], 'smoothness_mean': [0.1184], 'compactness_mean': [0.2776],
        'concavity_mean': [0.3001], 'concave points_mean': [0.1471], 'symmetry_mean': [0.2419],
        'fractal_dimension_mean': [0.07871], 'radius_se': [1.095], 'texture_se': [0.9053],
        'perimeter_se': [8.589], 'area_se': [153.4], 'smoothness_se': [0.006399],
        'compactness_se': [0.04904], 'concavity_se': [0.05373], 'concave points_se': [0.01587],
        'symmetry_se': [0.03003], 'fractal_dimension_se': [0.006193], 'radius_worst': [25.38],
        'texture_worst': [17.33], 'perimeter_worst': [184.6], 'area_worst': [2019],
        'smoothness_worst': [0.1622], 'compactness_worst': [0.6656], 'concavity_worst': [0.7119],
        'concave points_worst': [0.2654], 'symmetry_worst': [0.4601], 'fractal_dimension_worst': [0.1189]
    }
    input_df = pd.DataFrame(example_data)
    
    prediction, probabilities = predict_new_data(model, input_df)
    log_prediction(run_id, prediction, probabilities)
    
    # Mostrar resultados
    diagnosis = "Benigno (0)" if prediction[0] == 0 else "Maligno (1)"
    print(f"\nResultado de la predicción:")
    print(f"Diagnóstico: {diagnosis}")
    print(f"Probabilidad Benigno: {probabilities[0][0]:.4f}")
    print(f"Probabilidad Maligno: {probabilities[0][1]:.4f}")

if __name__ == "__main__":
    run_pipeline()
# %%
