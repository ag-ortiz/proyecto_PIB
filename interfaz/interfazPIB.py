
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PIL import Image
import sys
import cv2
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, disk
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, find_contours
from skimage.draw import polygon
from PyQt5.QtGui import QImage, QPixmap
from MachineLearning import train_models, evaluate_user_image, extract_features
class Ui_tcPIB(object):
    def setupUi(self, tcPIB):
        tcPIB.setObjectName("tcPIB")
        tcPIB.resize(1600, 1200)  # Incrementar el tamaño de la ventana
        tcPIB.setToolTipDuration(0)
        tcPIB.setStyleSheet("background-color : rgb(229, 251, 255)\n"
                            "")
        tcPIB.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(tcPIB)
        self.centralwidget.setObjectName("centralwidget")
        self.Titulo = QtWidgets.QTextEdit(self.centralwidget)
        self.Titulo.setGeometry(QtCore.QRect(24, 0, 1500, 87))  # Ajustar el tamaño del título
        font = QtGui.QFont()
        font.setFamily("Stylus BT")
        font.setPointSize(28)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.Titulo.setFont(font)
        self.Titulo.setToolTipDuration(0)
        self.Titulo.setStyleSheet("font: 28pt \"Stylus BT\";")
        self.Titulo.setLocale(QtCore.QLocale(QtCore.QLocale.Spanish, QtCore.QLocale.Argentina))
        self.Titulo.setFrameShape(QtWidgets.QFrame.Box)
        self.Titulo.setLineWidth(0)
        self.Titulo.setObjectName("Titulo")
        self.Titulo.setReadOnly(True)
        self.instrucciones = QtWidgets.QTextEdit(self.centralwidget)
        self.instrucciones.setGeometry(QtCore.QRect(24, 60, 853, 49))
        self.instrucciones.setFrameShape(QtWidgets.QFrame.Box)
        self.instrucciones.setLineWidth(0)
        self.instrucciones.setObjectName("instrucciones")
        self.instrucciones.setReadOnly(True)
        self.seleccionArchivo = QtWidgets.QPushButton(self.centralwidget)
        self.seleccionArchivo.setGeometry(QtCore.QRect(24, 108, 169, 25))
        self.seleccionArchivo.setStyleSheet("background-color:rgb(193, 238, 255);\n"
                                            "font: 10pt \"SuperFrench\";")
        self.seleccionArchivo.setObjectName("seleccionArchivo")
        self.muestraArchivo = QtWidgets.QLineEdit(self.centralwidget)
        self.muestraArchivo.setGeometry(QtCore.QRect(204, 108, 529, 25))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.muestraArchivo.sizePolicy().hasHeightForWidth())
        self.muestraArchivo.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.muestraArchivo.setFont(font)
        self.muestraArchivo.setStyleSheet("font: 8pt \"MS Shell Dlg 2\";")
        self.muestraArchivo.setFrame(False)
        self.muestraArchivo.setObjectName("muestraArchivo")
        self.ejecucionAlgortimo = QtWidgets.QPushButton(self.centralwidget)
        self.ejecucionAlgortimo.setGeometry(QtCore.QRect(24, 144, 169, 25))
        self.ejecucionAlgortimo.setStyleSheet("background-color:rgb(193, 238, 255);\n"
                                              "font: 10pt \"SuperFrench\";")
        self.ejecucionAlgortimo.setObjectName("ejecucionAlgortimo")
        self.ejecucionDiagnostico = QtWidgets.QPushButton(self.centralwidget)
        self.ejecucionDiagnostico.setGeometry(QtCore.QRect(200, 144, 169, 25))
        self.ejecucionDiagnostico.setStyleSheet("background-color:rgb(193, 238, 255);\n"
                                                "font: 10pt \"SuperFrench\";")
        self.ejecucionDiagnostico.setObjectName("ejecucionDiagnostico")
        self.textoDiagnostico = QtWidgets.QTextEdit(self.centralwidget)
        self.textoDiagnostico.setGeometry(QtCore.QRect(24, 228, 1557, 61))  # Ajustar el tamaño del texto de diagnóstico
        self.textoDiagnostico.setStyleSheet("background-color:rgb(193, 238, 255)")
        self.textoDiagnostico.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textoDiagnostico.setLineWidth(1)
        self.textoDiagnostico.setObjectName("textoDiagnostico")
        self.cuadroDiagnostico = QtWidgets.QTextEdit(self.centralwidget)
        self.cuadroDiagnostico.setGeometry(QtCore.QRect(24, 192, 1557, 37))  # Ajustar el tamaño del cuadro de diagnóstico
        self.cuadroDiagnostico.setFrameShape(QtWidgets.QFrame.Box)
        self.cuadroDiagnostico.setLineWidth(0)
        self.cuadroDiagnostico.setObjectName("cuadroDiagnostico")
        self.cuadroDiagnostico.setReadOnly(True)
        self.imOriginal = QtWidgets.QLabel(self.centralwidget)
        self.imOriginal.setGeometry(QtCore.QRect(24, 312, 517, 517))
        self.imOriginal.setStyleSheet("background-color:rgb(149, 204, 255)")
        self.imOriginal.setText("")
        self.imOriginal.setObjectName("imOriginal")
        self.imSegmentada = QtWidgets.QLabel(self.centralwidget)
        self.imSegmentada.setGeometry(QtCore.QRect(551, 312, 517, 517))
        self.imSegmentada.setStyleSheet("background-color:rgb(149, 204, 255)")
        self.imSegmentada.setText("")
        self.imSegmentada.setObjectName("imSegmentada")
        # Nuevo QLabel para mostrar otra imagen con espacio
        self.imNodulo = QtWidgets.QLabel(self.centralwidget)
        self.imNodulo.setGeometry(QtCore.QRect(1078, 312, 517, 517))
        self.imNodulo.setStyleSheet("background-color:rgb(149, 204, 255)")
        self.imNodulo.setText("")
        self.imNodulo.setObjectName("imNodulo")
        tcPIB.setCentralWidget(self.centralwidget)
        self.Menu_archivo = QtWidgets.QMenuBar(tcPIB)
        self.Menu_archivo.setGeometry(QtCore.QRect(0, 0, 1600, 26))  # Ajustar el tamaño de la barra de menú
        self.Menu_archivo.setObjectName("Menu_archivo")
        self.menuArchivo = QtWidgets.QMenu(self.Menu_archivo)
        self.menuArchivo.setObjectName("menuArchivo")
        self.menuGuardar_como = QtWidgets.QMenu(self.menuArchivo)
        self.menuGuardar_como.setObjectName("menuGuardar_como")
        tcPIB.setMenuBar(self.Menu_archivo)
        self.statusbar = QtWidgets.QStatusBar(tcPIB)
        self.statusbar.setObjectName("statusbar")
        tcPIB.setStatusBar(self.statusbar)
        self.actionPNG = QtWidgets.QAction(tcPIB)
        self.actionPNG.setObjectName("actionPNG")
        self.action_txt = QtWidgets.QAction(tcPIB)
        self.action_txt.setObjectName("action_txt")
        self.actionAbrir = QtWidgets.QAction(tcPIB)
        self.actionAbrir.setObjectName("actionAbrir")
        self.menuGuardar_como.addSeparator()
        self.menuGuardar_como.addAction(self.actionPNG)
        self.menuGuardar_como.addAction(self.action_txt)
        self.menuArchivo.addAction(self.actionAbrir)
        self.menuArchivo.addAction(self.menuGuardar_como.menuAction())
        self.Menu_archivo.addAction(self.menuArchivo.menuAction())
        self.resetButton = QtWidgets.QPushButton(self.centralwidget)
        self.resetButton.setGeometry(QtCore.QRect(380, 144, 169, 25))
        self.resetButton.setStyleSheet("background-color:rgb(193, 238, 255);\n"
                                       "font: 10pt \"SuperFrench\";")
        self.resetButton.setObjectName("resetButton")
        self.resetButton.setText("Resetear")
        self.resetButton.clicked.connect(self.resetear_valores)
        self.retranslateUi(tcPIB)
        QtCore.QMetaObject.connectSlotsByName(tcPIB)
        # Asignación de funciones a los botones y acciones
        self.seleccionArchivo.clicked.connect(self.seleccionar_archivo)
        self.ejecucionAlgortimo.clicked.connect(self.ejecutar_algoritmo)
        self.ejecucionDiagnostico.clicked.connect(self.ejecutar_diagnostico)
        self.actionAbrir.triggered.connect(self.seleccionar_archivo)
        self.actionPNG.triggered.connect(self.savePNG)
        self.action_txt.triggered.connect(self.saveTXT)
        # Entrenar los modelos y obtener el scaler
        self.models, self.scaler = train_models()
    def resetear_valores(self):
        self.muestraArchivo.clear()
        self.imOriginal.clear()
        self.imSegmentada.clear()
        self.imNodulo.clear()
        self.textoDiagnostico.clear()
    def retranslateUi(self, tcPIB):
        _translate = QtCore.QCoreApplication.translate
        tcPIB.setWindowTitle(_translate("tcPIB", "Tomografía Computada de Perfusión para Imágenes Biomédicas"))
        self.Titulo.setHtml(_translate("tcPIB", "<p align=\"center\"><span style=\" font-size:36pt;\">Tomografía Computada de Perfusión para Imágenes Biomédicas</span></p>"))
        self.instrucciones.setHtml(_translate("tcPIB", "<p align=\"center\"><span style=\" font-size:14pt;\">Seleccione una imagen para analizar y ejecute el algoritmo.</span></p>"))
        self.seleccionArchivo.setText(_translate("tcPIB", "Seleccionar archivo"))
        self.ejecucionAlgortimo.setText(_translate("tcPIB", "Ejecutar algoritmo"))
        self.ejecucionDiagnostico.setText(_translate("tcPIB", "Ejecutar diagnóstico"))
        self.menuArchivo.setTitle(_translate("tcPIB", "Archivo"))
        self.menuGuardar_como.setTitle(_translate("tcPIB", "Guardar como"))
        self.actionPNG.setText(_translate("tcPIB", "PNG"))
        self.action_txt.setText(_translate("tcPIB", ".txt"))
        self.actionAbrir.setText(_translate("tcPIB", "Abrir"))
    def seleccionar_archivo(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ReadOnly
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Seleccionar imagen", "", "All Files (*);;PNG Files (*.png);;JPEG Files (*.jpg;*.jpeg)", options=options)
        if filePath:
            self.muestraArchivo.setText(filePath)
            pixmap = QtGui.QPixmap(filePath)
            self.imOriginal.setPixmap(pixmap.scaled(self.imOriginal.size(), QtCore.Qt.KeepAspectRatio))
    def ejecutar_algoritmo(self):
        filePath = self.muestraArchivo.text()
        if not filePath:
            self.textoDiagnostico.setPlainText("Por favor, seleccione un archivo antes de ejecutar el algoritmo.")
            return
        try:
            original_image = Image.open(filePath)
            image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
            # Aplicar CLAHE
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(image)
            # Aplicar TV denoising
            tv_denoised_img = denoise_tv_chambolle(clahe_img, weight=2)
            # Calcular el umbral de Otsu
            otsu_threshold = threshold_otsu(tv_denoised_img)
            # Binarizar la imagen usando el umbral de Otsu
            binary_img = tv_denoised_img > otsu_threshold
            # Definir el tamaño del elemento estructurante para la erosión
            radius = 2
            selem = disk(radius)
            # Aplicar la operación de erosión a la imagen binarizada
            eroded_img = binary_erosion(binary_img, selem)
            # Restar las imágenes binarias
            subtracted_img = np.logical_xor(binary_img, eroded_img)
            # Eliminar las regiones conectadas al borde de la imagen
            cleaned_subtracted_img = clear_border(subtracted_img)
            # Etiquetar las regiones en la imagen limpiada
            labeled_img, num_labels = label(cleaned_subtracted_img, connectivity=1, return_num=True)
            # Mantener solo los bordes de los pulmones
            lung_borders = np.zeros_like(subtracted_img)
            for region in regionprops(labeled_img):
                if region.area > 100:
                    for coord in region.coords:
                        lung_borders[coord[0], coord[1]] = 1
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
                contour = contour.astype(int)
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
                points_flipped = np.flipud(points)
                mask = np.zeros_like(clahe_img, dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.concatenate([points, points_flipped]), 1)
                mask = np.flipud(mask)
                mask = np.rot90(mask, 2)
                convex_hull_values_rotated_img[mask.astype(bool)] = clahe_img[mask.astype(bool)]
            fig, ax = plt.subplots()
            ax.imshow(convex_hull_values_rotated_img, cmap='gray')
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            pixmap_segmented = QtGui.QPixmap('output_image.png')
            self.imSegmentada.setPixmap(pixmap_segmented.scaled(self.imSegmentada.size(), QtCore.Qt.KeepAspectRatio))
            # Mostrar la imagen del nódulo segmentado (asumiendo que se guarda en 'nodule_image.png')
            plt.imshow(largest_lung_borders, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('nodule_image.png', bbox_inches='tight', pad_inches=0)
            plt.close()
            pixmap_nodule = QtGui.QPixmap('nodule_image.png')
            self.imNodulo.setPixmap(pixmap_nodule.scaled(self.imNodulo.size(), QtCore.Qt.KeepAspectRatio))
            self.textoDiagnostico.setPlainText("Algoritmo ejecutado correctamente.")
        except Exception as e:
            self.textoDiagnostico.setPlainText(f"Error al ejecutar el algoritmo: {str(e)}")
    def ejecutar_diagnostico(self):
        filePath = self.muestraArchivo.text()
        if not filePath:
            self.textoDiagnostico.setPlainText("Por favor, seleccione un archivo antes de ejecutar el diagnóstico.")
            return
        try:
            im_segmentada, im_nodulo, predictions = evaluate_user_image(filePath, self.models, extract_features, self.scaler)
            if im_segmentada is not None and im_nodulo is not None:
                segmentada_qimage = QImage(im_segmentada.data, im_segmentada.shape[1], im_segmentada.shape[0], QImage.Format_Grayscale8)
                nodulo_qimage = QImage(im_nodulo.data, im_nodulo.shape[1], im_nodulo.shape[0], QImage.Format_Grayscale8)
                
                self.imSegmentada.setPixmap(QPixmap.fromImage(segmentada_qimage).scaled(self.imSegmentada.size(), QtCore.Qt.KeepAspectRatio))
                self.imNodulo.setPixmap(QPixmap.fromImage(nodulo_qimage).scaled(self.imNodulo.size(), QtCore.Qt.KeepAspectRatio))
            resultado = "Resultados del diagnóstico:\n\n"
            for model, probs in predictions.items():
                resultado += f"Modelo: {model}\n"
                resultado += f"Probabilidad de ser benigno: {probs['Probabilidad de ser benigno']:.2f}\n"
                resultado += f"Probabilidad de ser maligno: {probs['Probabilidad de ser maligno']:.2f}\n\n"
            self.textoDiagnostico.setPlainText(resultado)
        except Exception as e:
            self.textoDiagnostico.setPlainText(f"Error al ejecutar el diagnóstico: {str(e)}")
    def savePNG(self):
        filePath, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Guardar imagen", "", "PNG Files (*.png)")
        if filePath:
            exporter = ImageExporter(self.imSegmentada.pixmap())
            exporter.export(filePath)
    def saveTXT(self):
        filePath, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Guardar texto", "", "Text Files (*.txt)")
        if filePath:
            with open(filePath, 'w') as file:
                file.write(self.textoDiagnostico.toPlainText())
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    tcPIB = QtWidgets.QMainWindow()
    ui = Ui_tcPIB()
    ui.setupUi(tcPIB)
    tcPIB.show()
    sys.exit(app.exec_())
