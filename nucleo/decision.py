import numpy as np

def identificar_imagen(pca_model, vector_prueba, k=3, umbral_rec=15000.0, umbral_id=8000.0, usar_mahalanobis=True):
    """
    Aplica la regla de decisión de 3 pasos e incluye la extensión de Puntos Extra:
    Clasificación con Distancia de Mahalanobis en el espacio PCA (dividiendo entre √λ_i).
    """
    y_test, x_rec, error_rec = pca_model.proyectar(vector_prueba, k)
    
    # 1. Validación de pertenencia al dominio
    if error_rec > umbral_rec:
        return {
            "estado": "RECHAZADO_DOMINIO",
            "mensaje": f"Rechazado: Error e(x) = {error_rec:.2f} supera el umbral de dominio.",
            "error_reconstruccion": float(error_rec),
            "coordenadas": y_test.tolist(),
            "distancias": {}
        }
    
    y_entrenamiento = pca_model.proyecciones_entrenamiento[:, :k]
    etiquetas_unicas = sorted(list(set(pca_model.etiquetas)))
    lambdas_k = pca_model.lam[:k]
    
    centroides = {}
    distancias_por_metodo = {}
    
    for persona in etiquetas_unicas:
        indices = [i for i, etiqueta in enumerate(pca_model.etiquetas) if etiqueta == persona]
        if indices:
            proyecciones_persona = y_entrenamiento[indices]
            centroides[persona] = np.mean(proyecciones_persona, axis=0)
            
            # Aplicar Mahalanobis (Punto Extra) o Euclidiana
            if usar_mahalanobis:
                diferencia = centroides[persona] - y_test
                varianza_componentes = np.maximum(lambdas_k, 1e-5)
                distancias_por_metodo[persona] = float(np.sqrt(np.sum((diferencia ** 2) / varianza_componentes)))
            else:
                distancias_por_metodo[persona] = float(np.linalg.norm(centroides[persona] - y_test))
            
    if not distancias_por_metodo:
        return {"estado": "ERROR", "mensaje": "Sin datos de entrenamiento suficientes."}

    mejor_persona = min(distancias_por_metodo, key=distancias_por_metodo.get)
    min_distancia = distancias_por_metodo[mejor_persona]
    
    distancias_individuales = np.linalg.norm(y_entrenamiento - y_test, axis=1)
    detalle_distancias = {
        f"{pca_model.etiquetas[i]}": float(distancias_individuales[i]) 
        for i in range(len(pca_model.etiquetas))
    }
    
    # 2. Validación de persona desconocida
    umbral_efectivo = (umbral_id * 0.15) if usar_mahalanobis else umbral_id
    if min_distancia > umbral_efectivo:
        return {
            "estado": "PERSONA_DESCONOCIDA",
            "mensaje": f"Desconocido: Distancia ({min_distancia:.2f}) supera el umbral.",
            "error_reconstruccion": float(error_rec),
            "distancia_minima": float(min_distancia),
            "coordenadas": y_test.tolist(),
            "distancias": detalle_distancias
        }
        
    # 3. Asignación exitosa
    return {
        "estado": "IDENTIFICADO",
        "etiqueta": mejor_persona,
        "error_reconstruccion": float(error_rec),
        "distancia": float(min_distancia),
        "coordenadas": y_test.tolist(),
        "distancias": detalle_distancias
    }
   