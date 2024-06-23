import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import cv2
import numpy as np
import os
from joblib import load
from PyQt5.QtGui import QImage, QPixmap

def cambiar_directorio():
    try:
        os.chdir('C:\\Users\\Galle\\Documents\\ITBA\\PIB')
        new_dir = os.getcwd()
        print("Nuevo directorio:", new_dir)
    except FileNotFoundError:
        print(f"Directorio especificado no encontrado: {np.e}")

def cargar_modelos():
    scaler = None
    svm_model = None
    rf_model = None
    knn_model = None
    
    try:
        scaler = load('scaler.joblib')
        svm_model = load('SVM_model.joblib')
        rf_model = load('RandomForest_model.joblib')
        knn_model = load('KNN_model.joblib')
    except FileNotFoundError:
        print("No se encontraron los archivos necesarios (scaler.joblib, SVM_model.joblib, Random_Forest_model.joblib, KNN_model.joblib).")

    return scaler, svm_model, rf_model, knn_model

cambiar_directorio()
scaler, svm_model, rf_model, knn_model = cargar_modelos()

if scaler is not None and svm_model is not None and rf_model is not None and knn_model is not None:
    print("Modelos cargados exitosamente.")
else:
    print("Error al cargar los modelos.")

def gaussian_kernel(image, sigma=1):
    return cv2.GaussianBlur(image, (0, 0), sigma)

def polynomial_kernel(image, degree=2, coef=1):
    poly_img = np.float32(image) ** degree + coef
    return poly_img

def laplacian_kernel(image, sigma=1):
    return cv2.Laplacian(image, cv2.CV_64F, ksize=3)

def extract_features(image):
    try:
        fixed_size = (369, 369)  # Ajusta esto a la resolución usada durante el entrenamiento
        if image.shape[:2] != fixed_size:
            image = cv2.resize(image, fixed_size, interpolation=cv2.INTER_AREA)
        
        gaussian = gaussian_kernel(image)
        polynomial = polynomial_kernel(image)
        laplacian = laplacian_kernel(image)

        features = []
        features.extend(gaussian.flatten())
        features.extend(polynomial.flatten())
        features.extend(laplacian.flatten())

        print(f"longitud de features testeo: {len(features)}")

        return np.array(features)
    except Exception as e:
        print(f"Error en la extracción de features: {e}")
        raise

def qimage_to_numpy(image):
    width = image.width()
    height = image.height()
    ptr = image.bits()
    ptr.setsize(image.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)  # Assuming the image has 4 channels (RGBA)
    return arr

def preprocess_image(image, scaler):
    if isinstance(image, QPixmap):
        image = image.toImage()
    if isinstance(image, QImage):
        img_array = qimage_to_numpy(image)
    else:
        img_array = image  # Asumimos que ya es un numpy array

    # Redimensionar la imagen a una resolución fija
    fixed_size = (369, 369)  # Ajusta esto a la resolución usada durante el entrenamiento
    img_resized = cv2.resize(img_array, fixed_size, interpolation=cv2.INTER_AREA)

    img_resized=image

    print(f"imagen resized {img_resized.shape}")


    features = extract_features(img_resized)
    print("bien extraido")
    features = np.reshape(features, (1, -1))

    features_scaled = scaler.transform(features)
    print(f"features shape: {features.shape}")

    return features_scaled

def predict_probabilities(features_scaled, svm_model, rf_model, knn_model):
    try:
        svm_prob = svm_model.predict_proba(features_scaled)[0]
        rf_prob = rf_model.predict_proba(features_scaled)[0]
        knn_prob = knn_model.predict_proba(features_scaled)[0]

        print(f"SVM Probabilities: {svm_prob}")
        print(f"RF Probabilities: {rf_prob}")
        print(f"KNN Probabilities: {knn_prob}")

        return svm_prob, rf_prob, knn_prob
    except Exception as e:
        print(f"Error en la predicción de probabilidades: {e}")
        raise

