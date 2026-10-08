import numpy as np

def identificar_imagen(pca_model, vector_prueba, k=3, umbral_rec=4500.0, umbral_id=2000.0):
    """
    Aplica la regla de decisión en 3 pasos descrita en el PDF:
    1. Reconstrucción y evaluación de pertenencia al dominio (e(x) > umbral_rec).
    2. Evaluación de persona desconocida (distancia > umbral_id).
    3. Asignación del vecino más cercano.
    """
    y_test, x_rec, error_rec = pca_model.proyectar(vector_prueba, k)
    
    # 1. Comprobar si pertenece al dominio
    if error_rec > umbral_rec:
        return {
            "estado": "RECHAZADO_DOMINIO",
            "mensaje": "La imagen no pertenece al dominio (no parece un rostro del sistema).",
            "error_reconstruccion": float(error_rec),
            "coordenadas": y_test.tolist()
        }
    
    # 2. Proyectar entrenamiento con k componentes
    y_entrenamiento = pca_model.proyecciones_entrenamiento[:, :k]
    
    # Distancias Euclidianas a cada vecino
    distancias = np.linalg.norm(y_entrenamiento - y_test, axis=1)
    idx_min = np.argmin(distancias)
    min_distancia = distancias[idx_min]
    
    if min_distancia > umbral_id:
        return {
            "estado": "PERSONA_DESCONOCIDA",
            "mensaje": "Rostro detectado, pero corresponde a una persona no registrada.",
            "error_reconstruccion": float(error_rec),
            "distancia_minima": float(min_distancia),
            "coordenadas": y_test.tolist()
        }
    
    # 3. Asignación exitosa
    return {
        "estado": "IDENTIFICADO",
        "etiqueta": pca_model.etiquetas[idx_min],
        "error_reconstruccion": float(error_rec),
        "distancia": float(min_distancia),
        "coordenadas": y_test.tolist()
    }


   