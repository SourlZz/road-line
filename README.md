# Procesamiento de Video para Detección de Líneas de Carretera

Este proyecto utiliza OpenCV para procesar un video y detectar líneas de carretera mediante técnicas de procesamiento de imágenes.

## Requisitos

Antes de ejecutar el script, instala las siguientes dependencias:

```bash
pip install opencv-python numpy matplotlib
```

## Uso

Ejecuta el script pasando la ruta del video como argumento:

```bash
python script.py video.mp4
```

## Funcionamiento del Script

1. **Carga el video** y obtiene sus propiedades (FPS, resolución).
2. **Aplica corrección Gamma** para mejorar el contraste.
3. **Define una región de interés (ROI)** para enfocarse en la carretera.
4. **Convierte el video a formato YUV** para manipular la luminancia.
5. **Ajusta el histograma acumulativo** para mejorar la visibilidad de las líneas.
6. **Convierte a HSV** y aplica filtros de color para detectar líneas blancas y amarillas.
7. **Acumula histogramas de los canales H, S y V** para su análisis posterior.
8. **Guarda el video procesado** con las líneas detectadas.
9. **Muestra el resultado en tiempo real**, permitiendo detener la ejecución con 'q'.

## Notas

- El histograma promedio de los canales HSV puede visualizarse descomentando la línea `histograma_promedio()` al final del script.
- El archivo de salida `video_procesado.mp4` contendrá el video con las líneas resaltadas.
- Ajusta los valores de los filtros en `inRange()` si es necesario mejorar la detección de líneas en diferentes condiciones de iluminación.

