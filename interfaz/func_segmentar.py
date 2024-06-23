import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, disk
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, find_contours
from skimage.draw import polygon

def segmentar_imagen(imagen_8bits):
    # Crear el objeto CLAHE con los parámetros deseados
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

    # Aplicar CLAHE a la imagen
    clahe_img = clahe.apply(imagen_8bits)

    # Aplicar TV denoising a la imagen
    tv_denoised_img = denoise_tv_chambolle(clahe_img, weight=2)

    # Calcular el umbral de Otsu
    otsu_threshold = threshold_otsu(tv_denoised_img)

    # Binarizar la imagen usando el umbral de Otsu
    binary_img = tv_denoised_img > otsu_threshold

    # Definir el tamaño del elemento estructurante para la erosión
    radius = 2  # Puedes ajustar este valor según tus necesidades

    # Crear el elemento estructurante para la erosión
    selem = disk(radius)

    # Aplicar la operación de erosión a la imagen binarizada
    eroded_img = binary_erosion(binary_img, selem)

    # Restar las imágenes binarias
    subtracted_img = np.logical_xor(binary_img, eroded_img)

    # Mostrar la imagen resultante
    plt.imshow(subtracted_img, cmap='gray')
    plt.title('Resultado de la resta')
    plt.axis('off')
    plt.show()

    # Eliminar las regiones conectadas al borde de la imagen
    cleaned_subtracted_img = clear_border(subtracted_img)
    
    # Mostrar la imagen después de la limpieza de bordes
    plt.imshow(cleaned_subtracted_img, cmap='gray')
    plt.title('Imagen después de la limpieza de bordes')
    plt.axis('off')
    plt.show()


    # Etiquetar las regiones en la imagen limpiada
    labeled_img, num_labels = label(cleaned_subtracted_img, connectivity=1, return_num=True)

    # Mostrar el número de etiquetas encontradas
    print(f"Número de etiquetas encontradas: {num_labels}")
    ###MODIFICACIONES A PARTIR DE AQUI
    # # Mantener solo los bordes de los pulmones
    # lung_borders = np.zeros_like(subtracted_img)
    # for region in regionprops(labeled_img):
    #     if region.area > 330:  # Ajusta el umbral del área según tus necesidades
    #         for coord in region.coords:
    #             lung_borders[coord[0], coord[1]] = 1

    # # Mostrar los bordes de los pulmones
    # plt.imshow(lung_borders, cmap='gray')
    # plt.title('Bordes de los pulmones')
    # plt.axis('off')
    # plt.show()
    lung_borders = cleaned_subtracted_img
    # Verificar si se encontraron bordes de los pulmones
    if not np.any(lung_borders):
        print("No se encontraron bordes de los pulmones.")
        return

    # Etiquetar las regiones en la imagen de los bordes de los pulmones
    labeled_lung_borders, num_labels = label(lung_borders, connectivity=1, return_num=True)

    # Calcular el área de cada región etiquetada
    region_areas = [region.area for region in regionprops(labeled_lung_borders)]

    # Obtener los índices de las dos regiones más grandes
    largest_indices = np.argsort(region_areas)[-2:]

    # Crear una nueva imagen que contenga solo las dos regiones más grandes
    largest_lung_borders = np.zeros_like(lung_borders, dtype=bool)
    for idx in largest_indices:
        largest_lung_borders |= labeled_lung_borders == idx + 1

    # Mostrar los dos bordes principales de los pulmones
    plt.imshow(largest_lung_borders, cmap='gray')
    plt.title('Dos bordes principales de los pulmones')
    plt.axis('off')
    plt.show()

    # Encontrar los contornos dentro de los bordes de los pulmones
    contours = find_contours(largest_lung_borders, 0.5)

    # Crear una nueva imagen en blanco para dibujar los contornos suavizados y los convex hulls
    smoothed_contour_img = np.zeros_like(largest_lung_borders, dtype=np.uint8)
    convex_hull_img = np.zeros_like(largest_lung_borders, dtype=np.uint8)

    # Aproximar los contornos para suavizarlos y convertirlos a un formato adecuado
    smooth_contours = [np.squeeze(cv2.approxPolyDP(contour.astype(np.int32), 0.5, True)) for contour in contours]

    # Inicializar una lista para almacenar las coordenadas giradas de los convex hulls
    convex_hull_points = []

    # Dibujar los contornos suavizados y calcular los convex hulls para cada pulmón
    for contour in smooth_contours:
        contour = contour.astype(int)  # Asegurarse de que los valores sean enteros

        # Calcular el convex hull y dibujarlo
        convex_hull = cv2.convexHull(contour, clockwise=False)

        # Girar los puntos del convex hull 90 grados hacia el otro lado
        convex_hull[:, 0, 0], convex_hull[:, 0, 1] = convex_hull[:, 0, 1], convex_hull[:, 0, 0].copy()
        convex_hull[:, 0, 0] = convex_hull_img.shape[1] - convex_hull[:, 0, 0] - 1

        # Almacenar las coordenadas giradas del convex hull
        convex_hull_points.append(convex_hull[:, 0])

        # Dibujar el contorno suavizado
        rr, cc = polygon(contour[:, 0], contour[:, 1], smoothed_contour_img.shape)
        smoothed_contour_img[rr, cc] = 1

    # Crear una máscara combinada de todos los convex hulls girados
    convex_hull_mask = np.zeros_like(convex_hull_img, dtype=bool)
    for points in convex_hull_points:
        rr, cc = polygon(points[:, 1], points[:, 0], convex_hull_mask.shape)
        convex_hull_mask[rr, cc] = True

    # Espejar los convex hulls a lo largo del eje y
    convex_hull_mask = np.flipud(convex_hull_mask)

    # Girar los convex hulls 180 grados hacia el otro lado
    convex_hull_mask = np.rot90(convex_hull_mask, 2)

    # Crear una nueva imagen para mostrar los valores originales dentro de los convex hulls girados
    convex_hull_values_rotated_img = np.zeros_like(clahe_img)

    # Iterar sobre los convex hulls girados
    for points in convex_hull_points:
        # Espejar los puntos a lo largo del eje y
        points_flipped = np.flipud(points)

        # Crear una máscara para la región dentro del convex hull girado
        mask = np.zeros_like(clahe_img, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.concatenate([points, points_flipped]), 1)

        # Espejar la máscara a lo largo del eje y
        mask = np.flipud(mask)

        # Girar la máscara 180 grados
        mask = np.rot90(mask, 2)

        # Copiar los valores originales de la imagen dentro del convex hull girado a la imagen de salida
        convex_hull_values_rotated_img[mask.astype(bool)] = clahe_img[mask.astype(bool)]

    # Mostrar los contornos suavizados y los convex hulls
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(smoothed_contour_img, cmap='gray')
    axes[0].set_title('Contornos suavizados')
    axes[0].axis('off')
    axes[1].imshow(convex_hull_values_rotated_img, cmap='gray')
    axes[1].set_title('Valores originales dentro de los convex hulls girados y espejados')
    axes[1].axis('off')
    plt.show()

imagen_path = 'C:\\Users\\Beatriz\\Desktop\\UPIBI\\ITBA_PIB\\interfaz\\proyecto_PIB\\xrays sin segmentar PNG\\imagen_010.png'  # Reemplaza con la ruta de tu imagen

imagen_8bits = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

segmentar_imagen(imagen_8bits)