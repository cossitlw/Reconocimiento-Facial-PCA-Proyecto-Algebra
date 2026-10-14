import io
import base64
from PIL import Image
import numpy as np

def Cargar_y_preprocesar_imagen(fuente_imagen, tamano=(64, 64)):
    """
    Carga una imagen (ruta, bytes o Base64), maneja transparencias, 
    la convierte a escala de grises, redimensiona y aplana a vector d = h * w.
    """
    try:
        if isinstance(fuente_imagen, str) and fuente_imagen.startswith('data:image'):
            encabezado, base64_data = fuente_imagen.split(',', 1)
            datos_decodificados = base64.b64decode(base64_data)
            imagen = Image.open(io.BytesIO(datos_decodificados))
        elif isinstance(fuente_imagen, bytes):
            imagen = Image.open(io.BytesIO(datos_decodificados))
        else:
            imagen = Image.open(fuente_imagen)

        # Si la imagen tiene transparencia (RGBA), unificar sobre fondo blanco
        if imagen.mode in ('RGBA', 'LA') or (imagen.mode == 'P' and 'transparency' in imagen.info):
            fondo_blanco = Image.new('RGBA', imagen.size, (255, 255, 255, 255))
            fondo_blanco.paste(imagen, (0, 0), imagen)
            imagen = fondo_blanco.convert('RGB')
        else:
            imagen = imagen.convert('RGB')

        # Escala de grises y redimensionamiento con alta calidad (LANCZOS)
        imagen_gris = imagen.convert('L').resize(tamano, Image.Resampling.LANCZOS)
        vector = np.array(imagen_gris).astype(float).ravel()
        return vector, imagen_gris
    except Exception as e:
        raise ValueError(f"Error al procesar la imagen: {str(e)}")

def vector_a_imagen_base64(vector, tamano=(64, 64)):
    """Convierte un vector de píxeles normalizado a una imagen base64 para la interfaz web."""
    try:
        img_arr = vector.reshape(tamano)
        img_min, img_max = img_arr.min(), img_arr.max()
        if img_max > img_min:
            img_arr = ((img_arr - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_arr = np.zeros(tamano, dtype=np.uint8)
        img = Image.fromarray(img_arr)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception:
        return ""