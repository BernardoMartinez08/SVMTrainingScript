import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib as save_module


def load_data():
    print("Cargando el conjunto de datos Iris...")
    # El conjunto de datos Iris es un conjunto de datos de clasificación de 3 clases
    # que contiene 4 características (longitud del sépalo, ancho del sépalo, longitud del pétalo, ancho del pétalo)
    # y 150 muestras (50 muestras por clase)

    iris_data = load_iris()
    features = iris_data.data[:, :2]
    labels = iris_data.target
    return iris_data, features, labels


def split_data(features, labels, test_size, random_state):
    print("Dividiendo el conjunto de datos en entrenamiento y prueba...")
    features_train, features_test, labels_train, labels_test = (
        train_test_split(features, labels, test_size=test_size, random_state=random_state))
    return features_train, features_test, labels_train, labels_test


def train_model(features, labels, kernel, C):
    print("Entrenando el modelo SVM...")
    svm_classifier = SVC(kernel=kernel, C=C)
    svm_classifier.fit(features, labels)
    return svm_classifier


def save_model(svm_classifier, output_file_path):
    print("Guardando el modelo...")
    save_module.dump(svm_classifier, output_file_path)
    print("Modelo guardado exitosamente en", output_file_path)


def plot_original_dataset(iris, X, y):
    print("Graficando el conjunto de datos original...")
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=iris.target_names[y], palette='Set1', edgecolor='k', s=100)
    plt.title("Conjunto de Datos Iris Original")
    plt.xlabel("Longitud del Sépalo (cm)")
    plt.ylabel("Ancho del Sépalo (cm)")
    plt.legend(title='Especies', loc='upper right')
    plt.grid(True)
    plt.show()


def plot_decision_boundary(iris, X, y, classifier):
    print("Graficando el hiperplano de separación...")
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=iris.target_names[y], palette='Set1', edgecolor='k', s=100)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Crear la malla para graficar el hiperplano
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Graficar el hiperplano de separación
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
    plt.title("Conjunto de Datos Iris con Hiperplano de Separación (SVM)")
    plt.xlabel("Longitud del Sépalo (cm)")
    plt.ylabel("Ancho del Sépalo (cm)")
    plt.legend(title='Especies', loc='upper right')
    plt.grid(True)
    plt.show()


def main():
    # Cargar el conjunto de datos Iris
    iris_data, features, labels = load_data()

    # Dividir el conjunto de datos en entrenamiento y prueba
    features_train, features_test, labels_train, labels_test = (
        split_data(features, labels, 0.2, 42))

    # Visualizar el conjunto de datos original
    plot_original_dataset(iris_data, features_train, labels_train)

    # Entrenar el modelo SVM
    # 1.0 es el valor por defecto de C que es el parámetro de regularización que nos ayuda a controlar el sobreajuste
    # C es el margen de tolerancia que se le da a los datos de entrenamiento
    # Un valor de C pequeño significa un margen grande y un valor de C grande significa un margen pequeño
    # Un margen grande significa que se permite un mayor número de errores de entrenamiento
    # Un margen pequeño significa que se permite un menor número de errores de entrenamiento es decir
    # que se ajusta más a los datos

    svm_classifier = train_model(features_train, labels_train, 'linear', 1.0)

    # Guardar el modelo entrenado en un archivo
    save_model(svm_classifier, "svm_classifier.pkl")

    # Visualizar el hiperplano de separación
    plot_decision_boundary(iris_data, features_train, labels_train, svm_classifier)


if __name__ == "__main__":
    main()
