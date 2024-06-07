import os
from PIL import Image

# Ruta de la carpeta de datos PNG
data_path = "./data"

# Ruta de la carpeta donde se guardarán los archivos JPEG
output_path = "./data_jpg"

# Verificar si la carpeta de salida existe, si no, crearla
if not os.path.exists(output_path):
    os.makedirs(output_path)

print("Inicio del proceso de conversión de archivos PNG a JPEG...")

# Recorre cada carpeta y archivo dentro de data_path
for root, dirs, files in os.walk(data_path):
    print(f"Explorando la carpeta: {root}")
    for file in files:
        print(f"Procesando archivo: {file}")
        if file.lower().endswith(".png"):
            # Construye la ruta de entrada y salida
            input_path = os.path.join(root, file)
            output_folder = os.path.relpath(root, data_path)
            output_folder_path = os.path.join(output_path, output_folder)
            output_path_jpeg = os.path.join(output_folder_path, file.replace(".png", ".jpg"))

            # Verifica si la carpeta de salida existe, si no, crearla
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            try:
                # Abre la imagen PNG y la guarda como JPEG
                img = Image.open(input_path)
                img.convert("RGB").save(output_path_jpeg, "JPEG")
                print(f"Archivo convertido y guardado como JPEG: {output_path_jpeg}")
            except Exception as e:
                print(f"Error al convertir el archivo: {e}")

print("Conversiones de archivos PNG a JPEG completadas.")
