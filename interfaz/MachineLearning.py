# MachineLearning.py

import cv2
import numpy as np
import glob
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

def gaussian_kernel(image, sigma=1):
    return cv2.GaussianBlur(image, (0, 0), sigma)

def polynomial_kernel(image, degree=2, coef=1):
    poly_img = np.float32(image) ** degree + coef
    return poly_img

def laplacian_kernel(image, sigma=1):
    return cv2.Laplacian(image, cv2.CV_64F, ksize=3)

def extract_features(image):
    gaussian = gaussian_kernel(image)
    polynomial = polynomial_kernel(image)
    laplacian = laplacian_kernel(image)

    features = []
    features.extend(gaussian.flatten())
    features.extend(polynomial.flatten())
    features.extend(laplacian.flatten())

    return features

def train_models():
    lung_images_path = 'C:\\Users\\Beatriz\\Desktop\\UPIBI\\ITBA_PIB\\interfaz\\proyecto_PIB\\xrays segmentados\\todo\\pulmones\\*.png'
    masks_path = 'C:\\Users\\Beatriz\\Desktop\\UPIBI\\ITBA_PIB\\interfaz\\proyecto_PIB\\xrays segmentados\\todo\\mascaras\\*.png'
    output_path = 'C:\\Users\\Beatriz\\Desktop\\UPIBI\\ITBA_PIB\\interfaz\\proyecto_PIB\\xrays segmentados\\todo\\nodulos\\*.png'
    labels_path = 'C:\\Users\\Beatriz\\Desktop\\UPIBI\\ITBA_PIB\\interfaz\\proyecto_PIB\\clinical_information\\CLNDAT_EN.txt'

    lung_images_files = sorted(glob.glob(lung_images_path))
    masks_files = sorted(glob.glob(masks_path))

    for lung_image_file, mask_file in zip(lung_images_files, masks_files):
        lung_image = cv2.imread(lung_image_file, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        assert lung_image.shape == mask.shape, f"La imagen {lung_image_file} y la máscara {mask_file} tienen dimensiones diferentes."

        segmented_nodule = cv2.bitwise_and(lung_image, lung_image, mask=mask)

    labels_df = pd.read_csv(labels_path, delimiter='\t', header=None)
    labels_df.columns = ['filename', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'label', 'col9', 'col10', 'col11', 'col12']
    labels_df['label'] = labels_df['label'].apply(lambda x: 1 if x == 'malignant' else 0)

    image_files = sorted(glob.glob(output_path))

    features = []
    labels = []

    for i, file in enumerate(image_files):
        if i < len(labels_df):
            image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image_features = extract_features(image)
                features.append(image_features)
                label = labels_df['label'].iloc[i]
                labels.append(label)
            else:
                print(f"Error reading image: {file}")

    features = np.array(features)
    labels = np.array(labels)

    if len(features) >= 50:
        random_indices = np.random.choice(len(features), size=50, replace=False)
        train_features = features[random_indices]
        train_labels = labels[random_indices]

        test_indices = [i for i in range(len(features)) if i not in random_indices]
        test_features = features[test_indices]
        test_labels = labels[test_indices]

        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

        models = {
            'SVM': SVC(probability=True),
            'Random Forest': RandomForestClassifier(),
            'KNN': KNeighborsClassifier()
        }

        for name, model in models.items():
            model.fit(train_features, train_labels)
            y_pred = model.predict(test_features)
            print(f'--- {name} ---')
            print(classification_report(test_labels, y_pred))
            print(f'Accuracy: {accuracy_score(test_labels, y_pred)}')

        return models, scaler
    else:
        print("No hay suficientes imágenes para seleccionar aleatoriamente 50 para el entrenamiento.")
        return None, None

def evaluate_user_image(user_image_path, models, extract_features, scaler):
    user_image = cv2.imread(user_image_path, cv2.IMREAD_GRAYSCALE)

    if user_image is not None:
        user_image_features = extract_features(user_image)
        user_image_features = scaler.transform([user_image_features])
        print(f"\nEvaluating user-provided image: {user_image_path}")

        predictions = {}
        for name, model in models.items():
            prob = model.predict_proba(user_image_features)[0]
            predictions[name] = {
                'Probabilidad de ser benigno': prob[0],
                'Probabilidad de ser maligno': prob[1]
            }
            print(f'--- {name} ---')
            print(f'Probabilidad de ser benigno: {prob[0]:.2f}')
            print(f'Probabilidad de ser maligno: {prob[1]:.2f}')
        
        return predictions
    else:
        print(f"No se pudo cargar la imagen proporcionada por el usuario: {user_image_path}")
        return None

if __name__ == "__main__":
    models, scaler = train_models()
    
    if models is not None and scaler is not None:
        user_image_path = "ruta/a/la/imagen/del/usuario.jpg"
        predictions = evaluate_user_image(user_image_path, models, extract_features, scaler)
        print(predictions)


