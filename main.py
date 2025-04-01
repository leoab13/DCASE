from utils.dataloader import AudioDataset
from torch.utils.data import DataLoader

# Ruta al archivo CSV con las características y etiquetas
csv_path = "data/processed/audio_features1.csv"

# Crear el dataset
dataset = AudioDataset(csv_file=csv_path)

# Crear el DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#Sirve para en caso que el nombre del aeropuerto no aparezca en nuestras tags poder ver los nombres que no aparecen en dicha lista
for idx in range(len(dataset)):
    sample = dataset[idx]
    if sample['label'].item() == -1:  # Si la etiqueta es -1
        print(f"Etiqueta -1 encontrada en índice {idx}: {dataset.data.iloc[idx]['filename']}")

# Iterar sobre los lotes de datos
for batch_idx, batch in enumerate(dataloader):
    # Imprimir las dimensiones de las características y etiquetas
    print(f"Lote {batch_idx + 1}:")
    print(f"  Dimensiones de las características: {batch['features'].shape}")
    print(f"  Etiquetas: {batch['label']}")
    
    # Si necesitas detenerte después de algunos lotes para depuración
    if batch_idx == 2:  # Por ejemplo, detenerse después de 3 lotes
        break