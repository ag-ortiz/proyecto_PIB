import cv2
import numpy as np
import glob
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib  # Para guardar y cargar modelos
import pickle
from imblearn.over_sampling import SMOTE

# Ruta de las imágenes y etiquetas
image_path = 'C:\\Users\\krake\\Documents\\ITBA\\PIB\\archivos_oficiales\\xrays segmentados\\todo\\nodulos\\*.png'
labels_path = 'C:\\Users\\krake\\Documents\\ITBA\\PIB\\archivos_oficiales\\clinical_information\\CLNDAT_EN.txt'

# Cargar etiquetas
labels_df = pd.read_csv(labels_path, delimiter='\t', header=None)
labels_df.columns = ['filename', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'label', 'col9', 'col10', 'col11', 'col12']

# Convertir etiquetas a 0 (benigno) y 1 (maligno)
labels_df['label'] = labels_df['label'].apply(lambda x: 1 if x == 'malignant' else 0)

# Obtener lista de archivos de imagen
image_files = sorted(glob.glob(image_path))

# Inicializar listas para almacenar las características y etiquetas
features = []
labels = []

# Definir kernels
def gaussian_kernel(image, sigma=1):
    return cv2.GaussianBlur(image, (0, 0), sigma)

def polynomial_kernel(image, degree=2, coef=1):
    poly_img = np.float32(image) ** degree + coef
    return poly_img

def laplacian_kernel(image, sigma=1):
    return cv2.Laplacian(image, cv2.CV_64F, ksize=3)

# Función para extraer características
def extract_features(image):
    gaussian = gaussian_kernel(image)
    polynomial = polynomial_kernel(image)
    laplacian = laplacian_kernel(image)

    features = []
    features.extend(gaussian.flatten())
    features.extend(polynomial.flatten())
    features.extend(laplacian.flatten())

    return np.array(features)

# Cargar imágenes y extraer características
for i, file in enumerate(image_files):
    if i < len(labels_df):
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            fixed_size = (369, 369)  # Ajusta esto a la resolución usada durante el entrenamiento
            img_resized = cv2.resize(image, fixed_size, interpolation=cv2.INTER_AREA)
            image_features = extract_features(image)
            features.append(image_features)
            label = labels_df['label'].iloc[i]
            labels.append(label)
        else:
            print(f"Error reading image: {file}")

features = np.array(features)
labels = np.array(labels)

# Verificar si hay suficientes imágenes
if len(features) >= 50:
    # Seleccionar aleatoriamente 50 imágenes para entrenamiento
    random_indices = np.random.choice(len(features), size=50, replace=False)
    train_features = features[random_indices]
    train_labels = labels[random_indices]

    # Dividir el resto de los datos en conjunto de prueba
    test_indices = [i for i in range(len(features)) if i not in random_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]

    # Aplicar SMOTE para balancear las clases en el conjunto de entrenamiento
    smote = SMOTE()
    train_features, train_labels = smote.fit_resample(train_features, train_labels)

    # Escalar características
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # Modelos
    models = {
        'SVM': SVC(probability=True, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(class_weight='balanced'),
        'KNN': KNeighborsClassifier()
    }
    
    trained_models = {}

    # Entrenar y evaluar modelos
    for name, model in models.items():
        model.fit(train_features, train_labels)
        y_pred = model.predict(test_features)
        print(f'--- {name} ---')
        print(classification_report(test_labels, y_pred, zero_division=0))
        print(f'Accuracy: {accuracy_score(test_labels, y_pred)}')
        # Guardar el modelo entrenado en un diccionario
        trained_models[name] = model

    # Guardar el scaler también si lo necesitas para escalar nuevas entradas en la interfaz
    scaler_file = 'C:\\Users\\krake\\Documents\\ITBA\\PIB\\scaler.joblib'  # Actualiza la ruta donde guardar el scaler
    joblib.dump(scaler, scaler_file)
    
    # Guardar los modelos entrenados usando joblib o pickle
    for name, model in trained_models.items():
        model_file = f'C:\\Users\\krake\\Documents\\ITBA\\PIB\\{name}_model.joblib'  # Actualiza la ruta donde guardar cada modelo
        joblib.dump(model, model_file)
    print("Modelos y scaler guardados correctamente.")
else:
    print("No hay suficientes imágenes para seleccionar aleatoriamente 50 para el entrenamiento.")

