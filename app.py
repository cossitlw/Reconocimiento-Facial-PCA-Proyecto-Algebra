import os
import shutil
from flask import Flask, render_template, request, jsonify, url_for
from nucleo.preproceso import Cargar_y_preprocesar_imagen, vector_a_imagen_base64
from nucleo.pca import PCAFacial
from nucleo.decision import identificar_imagen

app = Flask(__name__)
modelo_pca = PCAFacial(tamano_imagen=(64, 64))

def cargar_datos_entrenamiento():
    carpeta_base = os.path.join(app.root_path, 'datos', 'entrenamiento')
    carpeta_static = os.path.join(app.root_path, 'static', 'celebridades')
    
    os.makedirs(carpeta_base, exist_ok=True)
    os.makedirs(carpeta_static, exist_ok=True)
    
    vectores = []
    etiquetas = []
    
    if not os.path.exists(carpeta_base):
        return

    for nombre_personaje in os.listdir(carpeta_base):
        ruta_personaje = os.path.join(carpeta_base, nombre_personaje)
        
        if os.path.isdir(ruta_personaje):
            foto_guardada = False
            for archivo in os.listdir(ruta_personaje):
                if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                    ruta_img = os.path.join(ruta_personaje, archivo)
                    
                    vec, _ = Cargar_y_preprocesar_imagen(ruta_img)
                    vectores.append(vec)
                    etiquetas.append(nombre_personaje)
                    
                    if not foto_guardada:
                        destino = os.path.join(carpeta_static, f"{nombre_personaje}.jpg")
                        shutil.copy(ruta_img, destino)
                        foto_guardada = True

    if len(vectores) >= 2:
        modelo_pca.entrenar(vectores, etiquetas)

@app.route('/')
def index():
    return render_template('index.html', entrenado=(modelo_pca.n >= 2))

@app.route('/api/pca_info', methods=['GET'])
def pca_info():
    if modelo_pca.n < 2:
        return jsonify({"error": "No hay suficientes imágenes cargadas en el modelo."}), 400
    
    eigen_base64 = [
        vector_a_imagen_base64(modelo_pca.W[:, i]) 
        for i in range(min(6, len(modelo_pca.lam)))
    ]
    
    return jsonify({
        "n_imagenes": modelo_pca.n,
        "dimension_original": modelo_pca.d,
        "eigenvalores": modelo_pca.lam.tolist(),
        "varianza_acumulada": modelo_pca.varianza_explicada_acumulada(),
        "imagen_promedio": vector_a_imagen_base64(modelo_pca.mu),
        "eigenfaces": eigen_base64
    })

@app.route('/identificar', methods=['POST'])
def identificar():
    try:
        data = request.get_json(silent=True) or {}
        imagen_data = data.get('imagen')
        k = int(data.get('k', 5))
        
        if not imagen_data:
            return jsonify({"error": "No se recibió imagen de la cámara."}), 400
            
        if modelo_pca.n < 2:
            return jsonify({"error": "El modelo PCA no tiene suficientes datos."}), 400

        vector_prueba, _ = Cargar_y_preprocesar_imagen(imagen_data)
        
        # Umbrales permisivos para encontrar el vecino más cercano en la base de datos
        resultado = identificar_imagen(modelo_pca, vector_prueba, k=k, umbral_rec=12000.0, umbral_id=10000.0)
        
        _, vec_rec, _ = modelo_pca.proyectar(vector_prueba, k)
        resultado["imagen_reconstruida"] = vector_a_imagen_base64(vec_rec)
        
        if "etiqueta" in resultado:
            personaje = resultado["etiqueta"]
            resultado["foto_celebridad"] = url_for('static', filename=f'celebridades/{personaje}.jpg')
            resultado["nombre_celebridad"] = personaje
            
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": f"Error procesando la imagen: {str(e)}"}), 400

if __name__ == '__main__':
    cargar_datos_entrenamiento()
    app.run(host='0.0.0.0', port=5000, debug=True)