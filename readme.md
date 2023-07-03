SVM 

Requisitos

Para ejecutar este código, asegúrate de tener las siguientes bibliotecas instaladas:

NumPy
Matplotlib
scikit-learn
emnist
TensorFlow (con Keras)

Puedes instalar las bibliotecas requeridas mediante el siguiente comando:
pip install numpy matplotlib scikit-learn emnist tensorflow

para usar importar las siguentes librerias

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from emnist import extract_training_samples, extract_test_samples
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
