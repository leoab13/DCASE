import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Cargar los datos
df = pd.read_csv("C:\\Users\\Leonardo\\Documents\\GitHub\\DCASE\\data\\processed\\features_labeled.csv")


X = df.drop(columns=["label"]).values  # Características
y = df["label"].values  # Etiquetas

# Normalizar características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Codificar etiquetas
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Remodelar los datos para LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Definir el modelo LSTM
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Definir el archivo de logs
log_file = "C:\\Users\\Leonardo\\Documents\\GitHub\\DCASE\\results\\training_log.txt"
with open(log_file, "w") as f:
    f.write("Epoch,Loss,Accuracy,Val_Loss,Val_Accuracy\n")

# Callback para guardar el historial de entrenamiento
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(log_file, "a") as f:
            f.write(f"{epoch + 1},{logs['loss']:.4f},{logs['accuracy']:.4f},{logs['val_loss']:.4f},{logs['val_accuracy']:.4f}\n")

# Callback de Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[CustomCallback(), early_stopping])

# Evaluar el modelo
y_pred = np.argmax(model.predict(X_test), axis=1)
report = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)

# Guardar métricas finales
metrics_file = "C:\\Users\\Leonardo\\Documents\\GitHub\\DCASE\\results\\final_metrics.txt"
with open(metrics_file, "w") as f:
    f.write("Precision, Recall, F1-Score\n")
    for label in encoder.classes_:
        f.write(f"{label},{report[label]['precision']:.4f},{report[label]['recall']:.4f},{report[label]['f1-score']:.4f}\n")

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("C:\\Users\\Leonardo\\Documents\\GitHub\\DCASE\\results\\confusion_matrix.png")
plt.close()

print("Entrenamiento finalizado. Resultados guardados en la carpeta 'results'.")
