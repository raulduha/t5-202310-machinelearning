

!pip install emnist

!pip install umap

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from emnist import extract_training_samples, extract_test_samples
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense

#extraemos emnist letters
train_images, train_labels = extract_training_samples('letters')
test_images, test_labels = extract_test_samples('letters')

#printeamos ejemplos sin procesar (para informe)
print("Ejemplos de imágenes sin procesar:")
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(train_images[i], cmap='gray')
    plt.title(chr(train_labels[i] + 96))  
    plt.axis('off')
plt.show()

#seleccionamos un subconjunto de datos para el entrensamiento
sample_size = 1000
train_indices = np.random.choice(len(train_images), size=sample_size, replace=False)
test_indices = np.random.choice(len(test_images), size=100, replace=False)
x_train_sample = train_images[train_indices]
y_train_sample = train_labels[train_indices]
x_test_sample = test_images[test_indices]
y_test_sample = test_labels[test_indices]

#normalizamos
x_train_sample = x_train_sample.reshape(sample_size, -1) / 255.0
x_test_sample = x_test_sample.reshape(len(test_indices), -1) / 255.0

# clasificador svc
svm_classifier = SVC()

#entrenamos y evaluamos con img sin procesar
svm_classifier.fit(x_train_sample, y_train_sample)
svm_raw_predictions = svm_classifier.predict(x_test_sample)
svm_raw_accuracy = accuracy_score(y_test_sample, svm_raw_predictions)

#reducimos dimensionabilidad con pca
pca = PCA(n_components=128)
x_train_pca = pca.fit_transform(x_train_sample)
x_test_pca = pca.transform(x_test_sample)

#entrenamos y evaluamos sv con pca
svm_classifier.fit(x_train_pca, y_train_sample)
svm_pca_predictions = svm_classifier.predict(x_test_pca)
svm_pca_accuracy = accuracy_score(y_test_sample, svm_pca_predictions)

# creamos  red convolucional simple para obtener características
input_shape = Input(shape=(28, 28))
flatten = Flatten()(input_shape)
output = Dense(128)(flatten)
embedding_model = Model(inputs=input_shape, outputs=output)
embedding_model.compile(optimizer='adam', loss='mse')
embedding_model.fit(x_train_sample.reshape(sample_size, 28, 28),
                    x_train_pca,
                    batch_size=32,
                    epochs=10,
                    verbose=0)

# extraemos caract. con embedding de la red simple
x_train_embedding = embedding_model.predict(x_train_sample.reshape(sample_size, 28, 28))
x_test_embedding = embedding_model.predict(x_test_sample.reshape(len(test_indices), 28, 28))

# entrenamos y evaliamos svm usando características de la red Simple
svm_classifier.fit(x_train_embedding, y_train_sample)
svm_embedding_predictions = svm_classifier.predict(x_test_embedding)
svm_embedding_accuracy = accuracy_score(y_test_sample, svm_embedding_predictions)

#callculamos accuracy por categoría
categories = np.unique(y_test_sample)
accuracy_per_category = []
for category in categories:
    category_indices = np.where(y_test_sample == category)[0]
    category_predictions = svm_embedding_predictions[category_indices]
    category_labels = y_test_sample[category_indices]
    category_accuracy = accuracy_score(category_labels, category_predictions)
    accuracy_per_category.append(category_accuracy)

# calculamos accuracy total
total_accuracy = accuracy_score(y_test_sample, svm_embedding_predictions)

#mostramos tabla
print("Accuracy por categoría:")
print("-----------------------")
for category, accuracy in zip(categories, accuracy_per_category):
    print(f"Categoría {category}: {accuracy}")

#mostramos accuracy total
print("\nAccuracy total:")
print(total_accuracy)

#grafico barras de accuracy por categoría
plt.bar(categories, accuracy_per_category)
plt.xlabel("Categoría")
plt.ylabel("Accuracy")
plt.title("Accuracy por categoría")
plt.show()

#grafico de barras de accuracy total
models = ["Imágenes sin procesar", "PCA (128 componentes)", "Red Simple (embedding)"]

accuracies = [svm_raw_accuracy, svm_pca_accuracy, svm_embedding_accuracy]
plt.bar(models, accuracies)
plt.xlabel("Modelo")
plt.ylabel("Accuracy")
plt.title("Accuracy total por modelo")
plt.show()

#printeamos algunas imagenes para informe
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(10):
    axes[i].imshow(train_images[i], cmap='gray')
    axes[i].set_title(f"Label: {train_labels[i]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

print(svm_raw_accuracy, svm_pca_accuracy, svm_embedding_accuracy)