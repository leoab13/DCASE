import pandas as pd

# Cargar el archivo CSV original
df = pd.read_csv("data/processed/all_audio_features.csv")

# Eliminar la columna 'folder'
df = df.drop(columns=["folder"])

# Diccionario de mapeo de etiquetas
label_mapping = {
    "airport": 0,
    "shopping_mall": 1,
    "metro_station": 2,
    "street_pedestrian": 3,
    "public_square": 4,
    "street_traffic": 5,
    "tram": 6,
    "bus": 7,
    "metro": 8,
    "park": 9
}

# Función para extraer la etiqueta desde 'filename'
def extract_label(filename):
    for key in label_mapping:
        if key in filename:
            return label_mapping[key]
    return -1  # Etiqueta desconocida

# Aplicar la extracción de etiquetas
df["label"] = df["filename"].apply(extract_label)

# Eliminar 'filename' después de extraer la etiqueta
df = df.drop(columns=["filename"])

# Verificar si hay etiquetas no asignadas (-1)
if (df["label"] == -1).sum() > 0:
    print("Algunas etiquetas no se asignaron correctamente. Revisa los nombres en 'filename'.")

# Guardar el nuevo archivo CSV
df.to_csv("processed/features_labeled.csv", index=False)

print("Archivo 'features_labeled.csv' guardado con éxito.")
