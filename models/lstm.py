import tensorflow as tf
import keras
import os


def load_dataset(dataset_url):
    """
    Descarga y carga el dataset como un objeto tf.data.Dataset.
    """
    dataset_path = keras.utils.get_file("dataset", dataset_url)

    # Aquí se debería incluir el código para cargar y procesar los archivos de audio
    dataset = tf.data.Dataset.list_files(os.path.join(os.path.dirname(dataset_path), "*.wav"))

    return dataset


def build_lstm_model(input_shape, num_classes):
    """
    Construye un modelo LSTM para clasificación de audio.
    """
    model = keras.models.Sequential([
        keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        keras.layers.LSTM(64),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def train_and_evaluate(model, train_dataset, val_dataset, epochs=10, log_file="training_log.txt"):
    """
    Entrena el modelo y guarda la salida en un archivo de texto.
    """
    with open(log_file, "w") as f:
        class LogCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                log_str = f"Epoch {epoch + 1}/{epochs}: " + ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()]) + "\n"
                f.write(log_str)
                print(log_str, end="")

        model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[LogCallback()])
