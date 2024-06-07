from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PIL import Image, ImageFilter
import sys

class Ui_tcPIB(object):
    
    def setupUi(self, tcPIB):
        tcPIB.setObjectName("tcPIB")
        tcPIB.resize(1107, 873)
        tcPIB.setToolTipDuration(0)
        tcPIB.setStyleSheet("background-color : rgb(229, 251, 255)\n"
"")
        tcPIB.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(tcPIB)
        self.centralwidget.setObjectName("centralwidget")
        self.Titulo = QtWidgets.QTextEdit(self.centralwidget)
        self.Titulo.setGeometry(QtCore.QRect(24, 0, 937, 87))
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
        self.textoDiagnostico = QtWidgets.QTextEdit(self.centralwidget)
        self.textoDiagnostico.setGeometry(QtCore.QRect(24, 228, 1057, 61))
        self.textoDiagnostico.setStyleSheet("background-color:rgb(193, 238, 255)")
        self.textoDiagnostico.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textoDiagnostico.setLineWidth(1)
        self.textoDiagnostico.setObjectName("textoDiagnostico")
        self.cuadroDiagnostico = QtWidgets.QTextEdit(self.centralwidget)
        self.cuadroDiagnostico.setGeometry(QtCore.QRect(24, 192, 1057, 37))
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
        self.imSegmentada.setGeometry(QtCore.QRect(552, 312, 529, 517))
        self.imSegmentada.setStyleSheet("background-color:rgb(149, 204, 255)")
        self.imSegmentada.setText("")
        self.imSegmentada.setObjectName("imSegmentada")
        tcPIB.setCentralWidget(self.centralwidget)
        self.Menu_archivo = QtWidgets.QMenuBar(tcPIB)
        self.Menu_archivo.setGeometry(QtCore.QRect(0, 0, 1107, 26))
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
        self.resetButton.setGeometry(QtCore.QRect(200, 144, 169, 25))
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
        self.actionAbrir.triggered.connect(self.seleccionar_archivo)
        self.actionPNG.triggered.connect(self.savePNG)
        self.action_txt.triggered.connect(self.saveTXT)
   
    def resetear_valores(self):
        # Limpiar la selección de archivo
        self.muestraArchivo.clear()
        
        # Limpiar las imágenes en imOriginal e imSegmentada
        self.imOriginal.clear()
        self.imSegmentada.clear()
        
        # Limpiar el texto del diagnóstico
        self.textoDiagnostico.clear()

    def seleccionar_archivo(self):
        # Función para manejar la acción de seleccionar un archivo
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        self.filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Seleccionar Archivo", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if self.filename:
            self.muestraArchivo.setText("Archivo seleccionado: " + self.filename)
            pixmap = QtGui.QPixmap(self.filename)
            self.imOriginal.setPixmap(pixmap.scaled(self.imOriginal.size(), QtCore.Qt.KeepAspectRatio))
            self.imSegmentada.clear()  # Limpiar imSegmentada

    def ejecutar_algoritmo(self):
        # Función para manejar la acción de ejecutar un algoritmo
        self.textoDiagnostico.setPlainText("Algoritmo ejecutado correctamente")
        if hasattr(self, 'filename') and self.filename:
            self.aplicar_filtro_mediana(self.filename)
        else:
            self.textoDiagnostico.setPlainText("Por favor, seleccione una imagen antes de ejecutar el algoritmo.")

    def aplicar_filtro_mediana(self, filename):
        # Función para aplicar el filtro de mediana a la imagen usando PIL
        imagen = Image.open(filename)
        if imagen is not None:
            self.imagen_mediana = imagen.filter(ImageFilter.MedianFilter(size=5))
            imagen_mediana_qt = self.convertir_imagen_qt(self.imagen_mediana)
            self.imSegmentada.setPixmap(imagen_mediana_qt.scaled(self.imSegmentada.size(), QtCore.Qt.KeepAspectRatio))
            self.textoDiagnostico.setPlainText("Algoritmo ejecutado correctamente y filtro de mediana aplicado.")
        else:
            self.textoDiagnostico.setPlainText("Error al cargar la imagen. Por favor, intente con otra imagen.")

    def convertir_imagen_qt(self, imagen):
        # Función para convertir una imagen PIL a QPixmap
        imagen_rgb = imagen.convert("RGB")
        data = imagen_rgb.tobytes("raw", "RGB")
        qimage = QtGui.QImage(data, imagen.size[0], imagen.size[1], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        return pixmap


    def saveTXT(self):
        texto_diagnostico = self.textoDiagnostico.toPlainText()
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Guardar Diagnóstico como TXT", "", "TXT Files (*.txt)")
        if filename:
                try:
                        # Abrir el archivo para escritura
                        file = open(filename, 'w')
                        # Escribir el texto en el archivo
                        file.write(texto_diagnostico)
                        # Cerrar el archivo
                        file.close()
                        self.textoDiagnostico.setPlainText("Diagnóstico guardado correctamente.")
                except Exception as e:
                # Manejar cualquier error
                        self.textoDiagnostico.setPlainText("Error al guardar el diagnóstico: " + str(e))

    def savePNG(self):
        if hasattr(self, 'imagen_mediana'):
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Guardar Imagen como PNG", "", "PNG Files (*.png);;All Files (*)", options=options)
            if file_path:
                self.imagen_mediana.save(file_path, "PNG")
                self.textoDiagnostico.setPlainText("Imagen guardada correctamente.")
            else:
                self.textoDiagnostico.setPlainText("Guardado cancelado.")
        else:
            self.textoDiagnostico.setPlainText("No hay imagen procesada para guardar.")

    def AbrirArchivo(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Abrir archivo de ECG", "", "PNG Files (*.png)")
        if filename:
            self.data = np.loadtxt(filename, delimiter=',', skiprows=1)
            self.update_graph()

    def retranslateUi(self, tcPIB):
        _translate = QtCore.QCoreApplication.translate
        tcPIB.setWindowTitle(_translate("tcPIB", "MainWindow"))
        self.Titulo.setHtml(_translate("tcPIB", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
        "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
        "p, li { white-space: pre-wrap; }\n"
        "</style></head><body style=\" font-family:\'Stylus BT\'; font-size:28pt; font-weight:400; font-style:normal;\">\n"
        "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">Clasificador tumoral de cáncer de pulmón</span></p></body></html>"))
        self.instrucciones.setHtml(_translate("tcPIB", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
        "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
        "p, li { white-space: pre-wrap; }\n"
        "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
        "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Stylus BT\'; font-size:12pt;\">Suba aqui la imagen correspondiente a la radiografía de toráx</span></p></body></html>"))
        self.seleccionArchivo.setText(_translate("tcPIB", "Seleccionar archivo"))
        self.muestraArchivo.setText(_translate("tcPIB", "Archivo seleccionado:"))
        self.ejecucionAlgortimo.setText(_translate("tcPIB", "Ejecutar algoritmo"))
        self.textoDiagnostico.setHtml(_translate("tcPIB", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
        "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
        "p, li { white-space: pre-wrap; }\n"
        "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
        "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.cuadroDiagnostico.setHtml(_translate("tcPIB", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
        "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
        "p, li { white-space: pre-wrap; }\n"
        "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
        "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Stylus BT\'; font-size:12pt;\">Diagnostico:</span></p></body></html>"))
        self.menuArchivo.setTitle(_translate("tcPIB", "Archivo"))
        self.menuGuardar_como.setTitle(_translate("tcPIB", "Guardar"))
        self.actionPNG.setText(_translate("tcPIB", "PNG"))
        self.action_txt.setText(_translate("tcPIB", "DIAGNOSTICO"))
        self.actionAbrir.setText(_translate("tcPIB", "Abrir"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    tcPIB = QtWidgets.QMainWindow()
    ui = Ui_tcPIB()
    ui.setupUi(tcPIB)
    tcPIB.show()
    sys.exit(app.exec_())