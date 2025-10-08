import requests
import numpy as np
from tensorflow import keras

# Charger une vraie image MNIST
(x_test, y_test) = keras.datasets.mnist.load_data()[1]
# Prendre la première image et la normaliser
test_image = x_test[0].astype("float32") / 255.0
test_image_flat = test_image.flatten().tolist()

# Données à envoyer
data = {
    "image": test_image_flat
}

# Envoyer la requête POST
try:
    response = requests.post(
        "http://localhost:5000/predict",
        headers={"Content-Type": "application/json"},
        json=data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Label attendu: {y_test[2]}")
        print(f"Prédiction: {result['prediction']}")
        print(f"Probabilités: {result['probabilities']}")
    else:
        print(f"Erreur: {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("Erreur: Impossible de se connecter à l'API. Assurez-vous qu'elle est lancée.")
except Exception as e:
    print(f"Erreur: {e}")