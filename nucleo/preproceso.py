import io
import base64
from PIL import Image
import numpy as np

def Cargar_y_preprocesar_imagen(fuente_imagen, tamano=(64, 64)):
    """
    Carga una imagen (ruta, bytes o Base64), la convierte a escala de grises,
    la redimensiona a 'tamano' (h, w) y la aplana a un vector 1D de d = h * w.
    """
    if isinstance(fuente_imagen, str) and fuente_imagen.startswith('data:image'):
        # Decodificación de imagen proveniente de la cámara (Base64)
        encabezado, base64_data = fuente_imagen.split(',', 1)
        datos_decodificados = base64.b64decode(base64_data)
        imagen = Image.open(io.BytesIO(datos_decodificados))
    elif isinstance(fuente_imagen, bytes):
        imagen = Image.open(io.BytesIO(fuente_imagen))
    else:
        imagen = Image.open(fuente_imagen)

    # Convertir a escala de grises y redimensionar
    imagen_gris = imagen.convert('L').resize(tamano, Image.Resampling.LANCZOS)
    
    # Convertir a arreglo NumPy y aplanar
    matriz_pixel = np.array(imagen_gris, dtype=float)
    vector_aplanado = matriz_pixel.ravel()
    
    return vector_aplanado, matriz_pixel

def vector_a_imagen_base64(vector, tamano=(64, 64)):
    """
    Convierte un vector reconstruido o una eigen-imagen a formato Base64 para mostrar en HTML.
    """
    matriz = vector.reshape(tamano)
    # Normalizar valores al rango 0-255
    matriz_norm = np.clip(matriz, 0, 255).astype(np.uint8)
    img = Image.fromarray(matriz_norm)
    
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    str_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{str_base64}"