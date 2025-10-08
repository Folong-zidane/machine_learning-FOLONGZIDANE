import tensorflow as tf
from tensorflow import keras
import numpy as np
import requests
import json

print("=" * 70)
print("DIAGNOSTIC COMPLET DU PIPELINE")
print("=" * 70)

# ===== PARTIE 1: Vérifier le modèle directement =====
print("\n[1] TEST DU MODÈLE EN LOCAL")
print("-" * 70)

# Charger le modèle
model = keras.models.load_model("mnist_model.h5")
print("✓ Modèle chargé")

# Charger les données
(x_test, y_test) = keras.datasets.mnist.load_data()[1]

# Test sur l'image 0
test_idx = 0
test_image_raw = x_test[test_idx]
test_label = y_test[test_idx]

print(f"\nImage test index: {test_idx}")
print(f"Label réel: {test_label}")
print(f"Shape raw: {test_image_raw.shape}")
print(f"Valeurs raw - Min: {test_image_raw.min()}, Max: {test_image_raw.max()}")

# Normaliser et prédire
test_image_normalized = test_image_raw.astype("float32") / 255.0
test_image_flat = test_image_normalized.reshape(1, 784)

print(f"\nAprès normalisation:")
print(f"Shape: {test_image_flat.shape}")
print(f"Min: {test_image_flat.min():.6f}, Max: {test_image_flat.max():.6f}")
print(f"Moyenne: {test_image_flat.mean():.6f}")

# Prédire
pred_local = model.predict(test_image_flat, verbose=0)
pred_class_local = np.argmax(pred_local[0])

print(f"\n✓ Prédiction locale: {pred_class_local}")
print(f"  Confiance: {pred_local[0][pred_class_local]:.4f}")
print(f"  Correct: {'OUI ✓' if pred_class_local == test_label else 'NON ✗'}")

# ===== PARTIE 2: Tester via l'API =====
print("\n" + "=" * 70)
print("[2] TEST VIA L'API")
print("-" * 70)

# Préparer les données pour l'API (MÊME IMAGE)
image_for_api = test_image_normalized.flatten().tolist()

print(f"\nDonnées envoyées à l'API:")
print(f"Type: {type(image_for_api)}")
print(f"Longueur: {len(image_for_api)}")
print(f"Premier pixel: {image_for_api[0]:.6f}")
print(f"Dernier pixel: {image_for_api[-1]:.6f}")
print(f"Min: {min(image_for_api):.6f}, Max: {max(image_for_api):.6f}")

# Envoyer à l'API
data = {"image": image_for_api}

try:
    response = requests.post(
        "http://localhost:5000/predict",
        headers={"Content-Type": "application/json"},
        json=data,
        timeout=5
    )
    
    if response.status_code == 200:
        result = response.json()
        pred_class_api = result['prediction']
        probs_api = result['probabilities'][0]
        
        print(f"\n✓ Réponse API reçue")
        print(f"  Prédiction API: {pred_class_api}")
        print(f"  Confiance: {probs_api[pred_class_api]:.4f}")
        print(f"  Correct: {'OUI ✓' if pred_class_api == test_label else 'NON ✗'}")
        
        # Comparer les probabilités
        print(f"\n[3] COMPARAISON DES PROBABILITÉS")
        print("-" * 70)
        print(f"{'Classe':<10} {'Local':<15} {'API':<15} {'Différence':<15}")
        print("-" * 70)
        
        max_diff = 0
        for i in range(10):
            diff = abs(pred_local[0][i] - probs_api[i])
            max_diff = max(max_diff, diff)
            marker = " ⚠" if diff > 0.01 else ""
            print(f"{i:<10} {pred_local[0][i]:<15.6f} {probs_api[i]:<15.6f} {diff:<15.6f}{marker}")
        
        print("-" * 70)
        print(f"Différence maximale: {max_diff:.6f}")
        
        if max_diff > 0.01:
            print("\n⚠ PROBLÈME: Les probabilités diffèrent significativement!")
            print("  → L'API ne traite PAS les données de la même manière que le test local")
            print("  → Vérifiez le preprocessing dans app.py")
        elif pred_class_local != pred_class_api:
            print("\n⚠ PROBLÈME: Même probabilités mais classes différentes (bizarre!)")
        else:
            print("\n✓ Tout est cohérent entre le test local et l'API")
            
    else:
        print(f"\n✗ Erreur HTTP {response.status_code}")
        print(f"Réponse: {response.text}")

except requests.exceptions.ConnectionError:
    print("\n✗ Impossible de se connecter à l'API")
    print("  Assurez-vous que app.py est lancé")
except Exception as e:
    print(f"\n✗ Erreur: {e}")

# ===== PARTIE 3: Test sur plusieurs images =====
print("\n" + "=" * 70)
print("[4] TEST SUR 5 IMAGES SUPPLÉMENTAIRES")
print("-" * 70)

for idx in [1, 2, 3, 4, 5]:
    img = x_test[idx].astype("float32") / 255.0
    img_flat = img.reshape(1, 784)
    
    # Local
    pred_local = model.predict(img_flat, verbose=0)
    pred_class_local = np.argmax(pred_local[0])
    
    # API
    try:
        response = requests.post(
            "http://localhost:5000/predict",
            json={"image": img.flatten().tolist()},
            timeout=5
        )
        pred_class_api = response.json()['prediction']
        
        match = "✓" if pred_class_local == pred_class_api else "✗"
        print(f"{match} Image {idx}: Vrai={y_test[idx]}, Local={pred_class_local}, API={pred_class_api}")
    except:
        print(f"✗ Image {idx}: Erreur API")
        break

print("\n" + "=" * 70)
print("FIN DU DIAGNOSTIC")
print("=" * 70)