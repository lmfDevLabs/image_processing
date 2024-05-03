import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def extraer_bordes(imagen_path, umbral1=50, umbral2=150):
    # Leer la imagen en escala de grises
    imagen = cv2.imread(imagen_path, 0)
    
    # Verificar si la imagen se ha cargado correctamente
    if imagen is None:
        print(f"No se pudo abrir la imagen: {imagen_path}")
        return

    # Aplicar una operación de desenfoque para reducir el ruido
    imagen_desenfocada = cv2.GaussianBlur(imagen, (5,5), 0)
    
    # Utilizar el algoritmo de Canny para la detección de bordes
    bordes = cv2.Canny(imagen_desenfocada, umbral1, umbral2)
    
    # Establecer el tamaño de la figura
    plt.figure(figsize=(15, 10))

    # Mostrar la imagen original y la imagen con los bordes detectados
    plt.subplot(121), plt.imshow(imagen, cmap='gray')
    plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(122), plt.imshow(bordes, cmap='gray')
    plt.title('Bordes Detectados'), plt.xticks([]), plt.yticks([])
    
    plt.show()

# Usar la función con la ruta de tu imagen
extraer_bordes('./data/DJI_0194.JPG')

def hough_transform(image_path):
    # Leer la imagen en escala de grises
    imagen = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Verificar si la imagen se ha cargado correctamente
    if imagen is None:
        print(f"No se pudo abrir la imagen: {image_path}")
        return
    #---------TESTING----------------------------
    # Suavizar la imagen usando un filtro gaussiano
    imagen_suavizada = cv2.GaussianBlur(imagen, (9, 9), 2)
    
    # Detectar los bordes usando el detector de bordes de Canny con umbrales ajustados
    bordes = cv2.Canny(imagen_suavizada, 50, 200, apertureSize=3)
    
    # Aplicar la Transformada de Hough con parámetros ajustados
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=20)
    
    # Crear una copia de la imagen original para dibujar las líneas
    imagen_lineas = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    
    # Dibujar las líneas en la imagen
    for linea in lineas:
        x1, y1, x2, y2 = linea[0]
        cv2.line(imagen_lineas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Convertir la imagen original y la imagen con líneas a RGB para visualización con matplotlib
    imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
    imagen_lineas = cv2.cvtColor(imagen_lineas, cv2.COLOR_BGR2RGB)
    
    # Establecer el tamaño de la figura
    plt.figure(figsize=(15, 10))
    
    # Mostrar la imagen original y la imagen con líneas detectadas
    plt.subplot(1, 2, 1)
    plt.imshow(imagen)
    plt.title('Imagen Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(imagen_lineas)
    plt.title('Transformada de Hough')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Usar la función con la ruta de tu imagen
hough_transform('./data/DJI_0194.JPG')

