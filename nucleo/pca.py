import numpy as np

class PCAFacial:
    def __init__(self, tamano_imagen=(64, 64)):
        self.tamano_imagen = tamano_imagen
        self.d = tamano_imagen[0] * tamano_imagen[1]
        self.mu = None
        self.Xc = None
        self.W = None
        self.lam = None
        self.n = 0
        self.etiquetas = []
        self.proyecciones_entrenamiento = None

    def entrenar(self, imagenes_vectores, etiquetas):
        """Ejecuta el PCA completo utilizando el truco de la matriz pequeña L (n x n)."""
        self.n = len(imagenes_vectores)
        if self.n < 2:
            raise ValueError("Se requieren al menos 2 imágenes para entrenar el modelo.")
        
        self.etiquetas = etiquetas
        X = np.stack(imagenes_vectores).astype(float)
        
        # 1. Imagen promedio
        self.mu = X.mean(axis=0)
        
        # 2. Centrado de datos
        self.Xc = X - self.mu
        
        # 3. Truco de la matriz reducida L (n x n)
        L = (self.Xc @ self.Xc.T) / (self.n - 1)
        
        # 4. Diagonalización con eigh (matriz simétrica)
        lam, V = np.linalg.eigh(L)
        
        # Ordenamiento descendente por eigenvalor
        orden = np.argsort(lam)[::-1]
        self.lam = lam[orden]
        V = V[:, orden]
        
        # Filtrado de eigenvalores numéricamente nulos o despreciables
        utiles = self.lam > (1e-10 * self.lam[0] if self.lam[0] > 0 else 1e-10)
        self.lam = self.lam[utiles]
        V = V[:, utiles]
        
        # 5. Obtención de Eigenfaces del espacio original (d dimensiones)
        self.W = self.Xc.T @ V
        self.W /= np.linalg.norm(self.W, axis=0)
        
        # Proyecciones del conjunto de entrenamiento
        self.proyecciones_entrenamiento = self.Xc @ self.W

    def proyectar(self, x, k):
        """Proyecta un vector x a k componentes, calcula la reconstrucción y el error e(x)."""
        k = min(k, len(self.lam))
        x_centrado = x - self.mu
        y = self.W[:, :k].T @ x_centrado
        x_rec = self.mu + self.W[:, :k] @ y
        error_rec = np.linalg.norm(x_centrado - self.W[:, :k] @ y)
        return y, x_rec, error_rec

    def varianza_explicada_acumulada(self):
        """Calcula la proporción de varianza explicada acumulada para k componentes."""
        varianza_total = np.sum(self.lam)
        if varianza_total == 0:
            return []
        return (np.cumsum(self.lam) / varianza_total).tolist()