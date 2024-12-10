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

if not cap.isOpened():
    print("Error al abrir el archivo de video")
    exit()

# Obtener las propiedades del video original
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Definir el codec y crear el objeto VideoWriter para guardar el video procesado
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video_procesado.mp4', fourcc, fps, (width, height))

# Variables para acumular los histogramas de cada canal (H, S, V)
hist_h_accum = np.zeros((256,), dtype=np.float32)
hist_s_accum = np.zeros((256,), dtype=np.float32)
hist_v_accum = np.zeros((256,), dtype=np.float32)
frame_count = 0

# Función para aplicar corrección Gamma
def ajustar_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    # Normalizar la imagen al rango [0, 1], aplicar la corrección gamma y volver a [0, 255]
    corrected_image = np.power(image / 255.0, invGamma) * 255.0
    return np.clip(corrected_image, 0, 255).astype(np.uint8)


# Función para acumular histogramas
def promedio_histogramas(h, s, v):
    global hist_h_accum, hist_s_accum, hist_v_accum, frame_count
    
    hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
    
    hist_h_accum += hist_h.flatten()
    hist_s_accum += hist_s.flatten()
    hist_v_accum += hist_v.flatten()
    
    frame_count += 1

# Función para mostrar el histograma promedio al final
def histograma_promedio():
    avg_hist_h = hist_h_accum / frame_count
    avg_hist_s = hist_s_accum / frame_count
    avg_hist_v = hist_v_accum / frame_count

    plt.figure(figsize=(12, 6))
    plt.plot(avg_hist_h, color='r', label='H (Matiz)')
    plt.plot(avg_hist_s, color='g', label='S (Saturación)')
    plt.plot(avg_hist_v, color='b', label='V (Brillo)')
    plt.title('Histogramas promedio en formato HSV')
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia promedio')
    plt.legend()
    plt.show()

# Definir la región de interés
def delimitar_imagen(image):
    height, width = image.shape[:2]
    top_y = int(0.6 * height)  # Comienza desde la mitad de la imagen
    bottom_y = height  # Hasta el borde inferior de la imagen
    mask = np.zeros_like(image)
    mask[top_y:bottom_y, 0:width] = image[top_y:bottom_y, 0:width]
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

#Histograma acumulativo
def histograma_acumulativo(imageY):
    canaly = imageY[:, :, 0]
    hist, bins = np.histogram(canaly.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalizado = cdf * hist.max() / cdf.max()

    for i in range(256):
        if i >= 240:  # Valores de blanco
            cdf_normalizado[i] = 255  # Máxima luminancia
        elif i >= 150:  # Valores de amarillo
            cdf_normalizado[i] = 200  # Luminancia alta para amarillo
        else:
            cdf_normalizado[i] = 0  # Oscurecer los demás

    luminancia_frame_ajustada = np.interp(canaly.flatten(), bins[:-1], cdf_normalizado)
    luminancia_frame_ajustada = luminancia_frame_ajustada.reshape(canaly.shape)
    imageY[:, :, 0] = luminancia_frame_ajustada.astype(np.uint8)
    return imageY

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Aplicar corrección Gamma
    frame_gamma = ajustar_gamma(frame, gamma=1.5)

    # Aplicar región de interés
    frame_roi = delimitar_imagen(frame_gamma)

    # Convertir a formato YUV para manipular el canal de luminancia
    lumniancia_frame = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2YUV)

    # Ajustar manualmente el histograma acumulado
    luminancia_ajustada = histograma_acumulativo(lumniancia_frame)

    # Convertir de nuevo a HSV para aplicar inRange
    hsv_frame = cv2.cvtColor(cv2.cvtColor(luminancia_ajustada, cv2.COLOR_YUV2BGR), cv2.COLOR_BGR2HSV)

    # Filtro para detectar líneas blancas
    lower_white = np.array([0, 0, 230], dtype=np.uint8)  # Elevar el valor mínimo de V para evitar brillos
    upper_white = np.array([180, 20, 255], dtype=np.uint8) # Rango de colores blancos
    mask_white = cv2.inRange(hsv_frame, lower_white, upper_white)

    # Filtro para detectar líneas amarillas
    lower_yellow = np.array([15, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    # Unir las dos máscaras
    mask_combined = cv2.bitwise_or(mask_white, mask_yellow)

    # Aplicar la máscara combinada
    result = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_combined)

    # Dividir los canales H, S y V
    h, s, v = cv2.split(hsv_frame)

    # Acumular histogramas de H, S, V
    promedio_histogramas(h, s, v)

    # Mostrar el resultado final
    cv2.imshow('Deteccion de lineas', result)
    # Escribir el frame procesado en el archivo de salida
    out.write(result)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#histograma_promedio()

cap.release()
cv2.destroyAllWindows()
