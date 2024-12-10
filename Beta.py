import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Manejo de argumentos para pasar el archivo de video
parser = argparse.ArgumentParser(description='Procesamiento de video para detectar líneas de carretera.')
parser.add_argument('video_path', type=str, help='Ruta del archivo de video')
args = parser.parse_args()

# Leer el video
cap = cv2.VideoCapture(args.video_path)

# Manda error si no se puede leer el video
if not cap.isOpened():
    print("Error al abrir el archivo de video")
    exit()

# Función para aplicar corrección Gamma
def ajustar_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    # Normalizar la imagen al rango [0, 1], aplicar la corrección gamma y volver a [0, 255]
    corrected_image = np.power(image / 255.0, invGamma) * 255.0
    return np.clip(corrected_image, 0, 255).astype(np.uint8)


# Función para mostrar el histograma de la imagen
def mostrar_histograma(image, title="Histograma"):
    plt.figure(figsize=(6, 4))
    
    if len(image.shape) == 2:  # Imagen en escala de grises o canal único
        plt.hist(image.ravel(), bins=256, range=[0, 256], color='gray')
    elif len(image.shape) == 3:  # Imagen en color, mostramos por canales (HSV o BGR)
        color = ('b', 'g', 'r')
        if image.shape[2] == 3:  # Caso de imagen HSV o BGR
            for i, col in enumerate(color):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
                plt.xlim([0, 256])
        else:  # Para otros casos no BGR o HSV
            plt.hist(image.ravel(), bins=256, range=[0, 256], color='gray')
    plt.title(title)
    plt.xlabel('Intensidad de píxeles')
    plt.ylabel('Frecuencia')
    plt.show()

# Definir la delimitación imagen
def delimitar_imagen(image):
    height, width = image.shape[:2]
    top_y = int(0.6 * height)  # Comienza desde la mitad de la imagen
    bottom_y = height  # Hasta el borde inferior de la imagen
    mask = np.zeros_like(image)
    mask[top_y:bottom_y, 0:width] = image[top_y:bottom_y, 0:width]
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Función para ajuste manual del histograma
def hisograma_promedio(imageY):
    canaly = imageY[:, :, 0]
    hist, bins = np.histogram(canaly.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalizado = cdf * hist.max() / cdf.max()
    
    for i in range(256):
        if i >= 240:
            cdf_normalizado[i] = 255
        elif i >= 150:
            cdf_normalizado[i] = 200
        else:
            cdf_normalizado[i] = 0

    luminancia_frame_ajustada = np.interp(canaly.flatten(), bins[:-1], cdf_normalizado)
    luminancia_frame_ajustada = luminancia_frame_ajustada.reshape(canaly.shape)
    imageY[:, :, 0] = luminancia_frame_ajustada.astype(np.uint8)
    return imageY

# Proceso de un solo frame
ret, frame = cap.read()
if ret:
    # 1. Mostrar el frame original
    cv2.imshow('Frame Original', frame)
    mostrar_histograma(frame, title="Histograma - Frame Original")
    cv2.waitKey(0)

    # 2. Aplicar corrección Gamma
    frame_gamma = ajustar_gamma(frame, gamma=1.5)
    cv2.imshow('Correccion Gamma', frame_gamma)
    mostrar_histograma(frame_gamma, title="Histograma - Correccion Gamma")
    cv2.waitKey(0)

    # 3. Aplicar delimitación imagen
    frame_roi = delimitar_imagen(frame_gamma)
    cv2.imshow('Región de Interés', frame_roi)
    mostrar_histograma(frame_roi, title="Histograma - delimitar imagen")
    cv2.waitKey(0)

    # 4. Convertir a formato YUV para manipular el canal de luminancia
    luminancia_frame = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2YUV)
    luminancia_ajustada = hisograma_promedio(luminancia_frame)
    frame_hist_adj = cv2.cvtColor(luminancia_ajustada, cv2.COLOR_YUV2BGR)
    cv2.imshow('Ajuste Manual de Histograma', frame_hist_adj)
    mostrar_histograma(luminancia_ajustada[:, :, 0], title="Histograma - Ajuste Manual")
    cv2.waitKey(0)

    # 5. Convertir a HSV para aplicar inRange
    hsv_frame = cv2.cvtColor(frame_hist_adj, cv2.COLOR_BGR2HSV)
    
    # Mostrar el histograma de los tres canales HSV
    plt.figure(figsize=(8, 4))
    for i, col in enumerate(['r', 'g', 'b']):  # r = H, g = S, b = V
        hist = cv2.calcHist([hsv_frame], [i], None, [256], [0, 256])
        plt.plot(hist, color=col, label=f'Canal {i+1}')
    plt.title('Histograma - Espacio HSV')
    plt.xlabel('Valor de píxeles')
    plt.ylabel('Frecuencia')
    plt.legend(['H (rojo)', 'S (verde)', 'V (azul)'])
    plt.show()

    # Filtro para detectar líneas blancas
    lower_white = np.array([0, 0, 230], dtype=np.uint8)
    upper_white = np.array([180, 20, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv_frame, lower_white, upper_white)
    cv2.imshow('Mascara de Blancos', mask_white)
    mostrar_histograma(mask_white, title="Histograma - Máscara de Blancos")
    cv2.waitKey(0)

    # Filtro para detectar líneas amarillas
    lower_yellow = np.array([0, 80, 80], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    cv2.imshow('Mascara de Amarillos', mask_yellow)
    mostrar_histograma(mask_yellow, title="Histograma - Máscara de Amarillos")
    cv2.waitKey(0)

    # 6. Unir las dos máscaras
    mask_combined = cv2.bitwise_or(mask_white, mask_yellow)
    cv2.imshow('Máscara Combinada (Blanco y Amarillo)', mask_combined)
    mostrar_histograma(mask_combined, title="Histograma - Mascara Combinada")
    cv2.waitKey(0)

    # 7. Aplicar la máscara combinada al frame
    result = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_combined)
    cv2.imshow('Resultado Final con Líneas Detectadas', result)
    mostrar_histograma(result, title="Histograma - Resultado Final")
    cv2.waitKey(0)
else:
    print("Error al leer el frame")

cap.release()
cv2.destroyAllWindows()
