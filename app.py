import os
import shutil
from flask import Flask, render_template, request, jsonify, url_for
from nucleo.preproceso import Cargar_y_preprocesar_imagen, vector_a_imagen_base64
from nucleo.pca import PCAFacial
from nucleo.decision import identificar_imagen

app = Flask(__name__)
modelo_pca = PCAFacial(tamano_imagen=(64, 64))

datos_entrenamiento_cache = []

def cargar_datos_entrenamiento():
    global datos_entrenamiento_cache
    datos_entrenamiento_cache = []
    carpeta_base = os.path.join(app.root_path, 'datos', 'entrenamiento')
    carpeta_static = os.path.join(app.root_path, 'static', 'celebridades')
    
    os.makedirs(carpeta_base, exist_ok=True)
    os.makedirs(carpeta_static, exist_ok=True)
    
    vectores = []
    etiquetas = []

    if os.path.exists(carpeta_base):
        for nombre_personaje in os.listdir(carpeta_base):
            ruta_personaje = os.path.join(carpeta_base, nombre_personaje)
            
            if os.path.isdir(ruta_personaje):
                foto_guardada = False
                for archivo in os.listdir(ruta_personaje):
                    if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                        ruta_img = os.path.join(ruta_personaje, archivo)
                        try:
                            vec, img_gris = Cargar_y_preprocesar_imagen(ruta_img)
                            vectores.append(vec)
                            etiquetas.append(nombre_personaje)
                            
                            img_b64 = vector_a_imagen_base64(vec)
                            datos_entrenamiento_cache.append({
                                "etiqueta": nombre_personaje,
                                "imagen": img_b64,
                                "archivo": archivo
                            })
                            
                            if not foto_guardada:
                                destino = os.path.join(carpeta_static, f"{nombre_personaje}.jpg")
                                shutil.copy(ruta_img, destino)
                                foto_guardada = True
                        except Exception as e:
                            print(f"Advertencia: No se pudo cargar {ruta_img}: {e}")

    if len(vectores) >= 2:
        modelo_pca.entrenar(vectores, etiquetas)

@app.route('/')
def index():
    cargar_datos_entrenamiento()
    return render_template('index.html', entrenado=(modelo_pca.n >= 2))

@app.route('/api/pca_info', methods=['GET'])
def pca_info():
    cargar_datos_entrenamiento()
    
    if modelo_pca.n < 2:
        return jsonify({"error": "No hay suficientes imágenes cargadas en el modelo (mínimo 2)."}), 400
    
    eigen_base64 = [
        vector_a_imagen_base64(modelo_pca.W[:, i]) 
        for i in range(min(6, len(modelo_pca.lam)))
    ]
    
    centradas_base64 = []
    if modelo_pca.Xc is not None:
        for i in range(min(3, modelo_pca.Xc.shape[0])):
            centradas_base64.append(vector_a_imagen_base64(modelo_pca.Xc[i]))

    L_matrix = []
    if modelo_pca.Xc is not None and modelo_pca.n > 0:
        L = (modelo_pca.Xc @ modelo_pca.Xc.T) / (modelo_pca.n - 1)
        l_min, l_max = L.min(), L.max()
        if l_max > l_min:
            L_norm = ((L - l_min) / (l_max - l_min) * 255).astype(int).tolist()
        else:
            L_norm = [[0]*modelo_pca.n for _ in range(modelo_pca.n)]
        L_matrix = L_norm

    return jsonify({
        "n_imagenes": modelo_pca.n,
        "dimension_original": modelo_pca.d,
        "eigenvalores": modelo_pca.lam.tolist(),
        "varianza_acumulada": modelo_pca.varianza_explicada_acumulada(),
        "imagen_promedio": vector_a_imagen_base64(modelo_pca.mu),
        "imagenes_centradas": centradas_base64,
        "eigenfaces": eigen_base64,
        "datos_entrenamiento": datos_entrenamiento_cache,
        "matriz_L": L_matrix
    })

@app.route('/identificar', methods=['POST'])
def identificar():
    try:
        cargar_datos_entrenamiento()
        
        data = request.get_json(silent=True) or {}
        imagen_data = data.get('imagen')
        k = int(data.get('k', 5))
        usar_mah = data.get('mahalanobis', True)
        
        if not imagen_data:
            return jsonify({"error": "No se recibió imagen para procesar."}), 400
            
        if modelo_pca.n < 2:
            return jsonify({"error": "El modelo PCA no está entrenado (faltan imágenes)."}), 400

        k = max(1, min(k, len(modelo_pca.lam)))
        vector_prueba, img_gris_prueba = Cargar_y_preprocesar_imagen(imagen_data)
        
        resultado = identificar_imagen(modelo_pca, vector_prueba, k=k, umbral_rec=12000.0, umbral_id=6000.0, usar_mahalanobis=usar_mah)
        
        _, vec_rec, error_rec = modelo_pca.proyectar(vector_prueba, k)
        resultado["imagen_reconstruida"] = vector_a_imagen_base64(vec_rec)
        resultado["imagen_preprocesada_prueba"] = vector_a_imagen_base64(vector_prueba)
        resultado["metrica_usada"] = "Mahalanobis" if usar_mah else "Euclidiana"
        
        # Enviar eigenvalores dinámicamente en cada escaneo
        resultado["eigenvalores"] = modelo_pca.lam.tolist()
        resultado["n_imagenes"] = modelo_pca.n
        resultado["dimension_original"] = modelo_pca.d
        
        if "etiqueta" in resultado:
            personaje = resultado["etiqueta"]
            resultado["foto_celebridad"] = url_for('static', filename=f'celebridades/{personaje}.jpg')
            resultado["nombre_celebridad"] = personaje
            
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": f"Error procesando la imagen: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)