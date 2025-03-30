import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os

#-----------------------------------
# Sección 0: Funciones para cargar datos reales
def extract_label_from_filename(filename):
    """
    Extrae la etiqueta del nombre del archivo.
    Se asume que el nombre sigue el formato:
      "airport-barcelona-0-0-0-a.wav"
    Donde se toma el segundo elemento (después del primer guion)
    como la etiqueta (por ejemplo, "barcelona").
    """
    parts = filename.split('-')
    if len(parts) >= 2:
        return parts[1]
    else:
        return "unknown"

def load_real_data(csv_path):
    """
    Carga datos reales desde un CSV con características de audio.
    
    Se asume que el CSV puede tener las siguientes columnas:
      - Para audio_features1.csv:
          [mfcc_1, mfcc_2, ..., mfcc_13, spec_contrast_1, ..., spec_contrast_6, 
           spec_bandwidth, entropy, rms, filename]
      - Para all_audio_features.csv:
          Lo mismo que anterior, más una columna 'folder'.
    
    Si no existe una columna 'label', se extrae la etiqueta desde 'filename'
    usando extract_label_from_filename. Luego se mapean las etiquetas únicas a números.
    
    Retorna:
      X: Matriz numpy de características (tipo float32).
      y: Vector numpy de etiquetas numéricas (tipo int32).
    """
    df = pd.read_csv(csv_path)
    print("Columnas del CSV:", df.columns.tolist())
    
    # Si no existe la columna 'label', se crea extrayéndola desde 'filename'
    if 'label' not in df.columns:
        df['label'] = df['filename'].apply(extract_label_from_filename)
        # Mapear las etiquetas únicas a números
        unique_labels = sorted(df['label'].unique())
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        df['label'] = df['label'].map(label_mapping)
        print("Etiquetas únicas mapeadas:", label_mapping)
    
    # Seleccionar columnas de características: todas menos 'filename', 'folder' y 'label'
    feature_columns = [col for col in df.columns if col not in ['filename', 'folder', 'label']]
    X = df[feature_columns].values.astype(np.float32)
    y = df['label'].values.astype(np.int32)
    return X, y

#-----------------------------------
# Sección 1: Creación del Modelo LSTM Simulado
def create_dummy_lstm(input_shape, num_classes=2):
    """
    Crea un modelo LSTM simulado para pruebas.
    """
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

#-----------------------------------
# Sección 2: Evaluación del Modelo
def evaluate_model(model, X, y):
    """
    Genera predicciones con el modelo y calcula las métricas de evaluación.
    """
    # Redimensionar X para que tenga forma (N, features, 1)
    X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Obtener predicciones
    y_pred_prob = model.predict(X_reshaped)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calcular métricas
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    cm = confusion_matrix(y, y_pred)
    
    print(f"Precisión: {acc:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("Matriz de confusión:")
    print(cm)

#-----------------------------------
# Sección 3: Lectura y Graficado del Log de Entrenamiento
def read_training_log(log_file):
    """
    Simula la lectura de un archivo de log de entrenamiento.
    Retorna listas de epochs, losses y accuracies.
    
    Cuando dispongas del log real, adapta esta función para parsear el archivo.
    Se espera que cada línea del log tenga el siguiente formato (ejemplo):
      "Epoch 1/10 - loss: 0.5462 - accuracy: 0.84"
    """
    # Datos simulados para 10 épocas:
    epochs = list(range(1, 11))
    losses = np.linspace(1.0, 0.3, num=10)       # Pérdida simulada
    accuracies = np.linspace(0.5, 0.95, num=10)    # Precisión simulada
    return epochs, losses, accuracies
#-----------------------------------
"""
def read_training_log(log_file):
    # Lee un archivo de log de entrenamiento y extrae las listas de epochs, losses y accuracies.
    # Se espera que cada línea del log tenga el siguiente formato (ejemplo):
    #  "Epoch 1/10 - loss: 0.5462 - accuracy: 0.84"
    epochs = []
    losses = []
    accuracies = []
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Verificar que la línea contiene la información esperada
            if line.startswith("Epoch") and "loss:" in line and "accuracy:" in line:
                try:
                    # Dividir la línea en partes
                    parts = line.split(" - ")
                    # parts[0] es "Epoch 1/10"
                    # parts[1] es "loss: 0.5462"
                    # parts[2] es "accuracy: 0.84"
                    epoch_str = parts[0].split()[1]   # "1/10"
                    epoch_num = int(epoch_str.split('/')[0])
                    loss_value = float(parts[1].split()[1])
                    accuracy_value = float(parts[2].split()[1])
                    
                    epochs.append(epoch_num)
                    losses.append(loss_value)
                    accuracies.append(accuracy_value)
                except Exception as e:
                    print(f"Error al parsear la línea: {line}. Error: {e}")
    return epochs, losses, accuracies

"""

def plot_training_results(epochs, losses, accuracies):
    """
    Grafica la pérdida y la precisión durante el entrenamiento.
    """
    plt.figure(figsize=(12, 6))
    
    # Gráfico de Loss (Pérdida)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, marker='o', label="Loss")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Pérdida durante el entrenamiento")
    plt.legend()
    
    # Gráfico de Accuracy (Precisión)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, marker='o', color="orange", label="Accuracy")
    plt.xlabel("Época")
    plt.ylabel("Precisión")
    plt.title("Precisión durante el entrenamiento")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

#-----------------------------------
# Sección Principal: Ejecución del Script
if __name__ == "__main__":
    # 1. Cargar datos reales desde el CSV (ya procesados en tareas anteriores)
    csv_path = "data/processed/all_audio_features.csv"  # O usar "audio_features1.csv"
    X, y = load_real_data(csv_path)
    print("Dimensiones de X:", X.shape)
    print("Dimensiones de y:", y.shape)
    
    # 2. Crear el modelo LSTM simulado
    # Cuando se disponga del modelo real, reemplazar la siguiente línea con:
    # model_lstm = load_model('models/model.h5')
    model_lstm = create_dummy_lstm(input_shape=(X.shape[1], 1)) # <--- Eliminar esta línea
    print("Modelo LSTM simulado creado.")
    
    # 3. Evaluar el modelo simulado con los datos reales
    evaluate_model(model_lstm, X, y)
    
    # 4. Lectura y graficado del log de entrenamiento
    log_file_path = "logs/training_log.txt"
    
    # ----- Líneas para usar el log real (descomentar cuando se disponga del log real) -----
    # if os.path.exists(log_file_path):
    #     epochs, losses, accuracies = read_training_log(log_file_path)  # Ajustar read_training_log() según el formato real
    # else:
    #     print("No se encontró el archivo de log de entrenamiento.")
    #     epochs, losses, accuracies = [], [], []
    # --------------------------------------------------------------------------------------
    
    # Por ahora, se usan datos simulados para el log:
    epochs, losses, accuracies = read_training_log(log_file_path) # <--- Cuando se tenga el log real, eliminar esta línea
    print(f"Log leído: {len(epochs)} épocas.")
    plot_training_results(epochs, losses, accuracies)

