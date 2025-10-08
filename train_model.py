import tensorflow as tf
from tensorflow import keras
import numpy as np
import mlflow
import mlflow.tensorflow

# Variables pour les paramètres
EPOCHS = 10
BATCH_SIZE = 128
DROPOUT_RATE = 0.2

# Chargement du jeu de données MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalisation des données
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Redimensionnement des images pour les réseaux fully-connected
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Lancement de la session de suivi MLflow
with mlflow.start_run():
    # Enregistrement des paramètres
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("dropout_rate", DROPOUT_RATE)
    
    # Construction du modèle
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.summary()
    print("Compilation du modèle")
    
    # Compilation du modèle
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entraînement du modèle
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )
    
    # Évaluation du modèle
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Précision sur les données de test : {test_acc:.4f}")
    print(f"Perte sur les données de test : {test_loss:.4f}")

    #test sur quelques images
    print("Test sur quelques images")
    indices = np.random.choice(len(x_test), 10, replace=False)
    correct = 0
    for idx in indices:
        img = x_test[idx:idx+1]
        pred = model.predict(img, verbose=0)
        pred_class = np.argmax(pred[0])
        true_class = y_test[idx]
        if pred_class == true_class:
            correct += 1
        status = "Correct" if pred_class == true_class else "Incorrect"
        print(f"Image {idx}: Vrai label: {true_class}, Prédiction: {pred_class}, Statut: {status} ")
    print(f" \n Predictions correctes sur 10 images : {correct}/10 \n")
        # Afficher les probabilités pour chaque classe
       
    # Enregistrement des métriques
    mlflow.log_metric("test_accuracy", test_acc)
    
    # Sauvegarde du modèle
    model.save("mnist_model.h5")
    print("Modèle sauvegardé sous mnist_model.h5")
    
    # Enregistrement du modèle complet
    mlflow.log_artifact("mnist_model.h5")