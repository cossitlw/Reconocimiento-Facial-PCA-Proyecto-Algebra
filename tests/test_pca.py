import pytest
import numpy as np
from nucleo.pca import PCAFacial

def test_pca_dimensiones():
    pca = PCAFacial(tamano_imagen=(8, 8))
    # Generar 5 vectores aleatorios
    vectores_falsos = [np.random.rand(64) for _ in range(5)]
    etiquetas = [f"persona_{i}" for i in range(5)]
    
    pca.entrenar(vectores_falsos, etiquetas)
    
    # Con 5 imágenes, hay a lo más 4 componentes útiles
    assert len(pca.lam) <= 4
    assert pca.W.shape[0] == 64
    assert pca.W.shape[1] == len(pca.lam)

def test_proyeccion_reconstruccion():
    pca = PCAFacial(tamano_imagen=(8, 8))
    vectores = [np.random.rand(64) for _ in range(5)]
    pca.entrenar(vectores, [f"img_{i}" for i in range(5)])
    
    y, x_rec, error = pca.proyectar(vectores[0], k=2)
    assert len(y) == 2
    assert x_rec.shape[0] == 64
    assert error >= 0.0