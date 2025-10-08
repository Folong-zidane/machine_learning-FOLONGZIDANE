# TP Deep Learning - Reconnaissance de Chiffres MNIST

## Description
Ce projet implémente un réseau de neurones pour la classification des chiffres manuscrits du dataset MNIST, avec suivi des expérimentations via MLflow et déploiement via une API Flask conteneurisée.

## Structure du projet
```
├── train_model.py      # Script d'entraînement du modèle
├── app.py             # API Flask pour servir le modèle
├── requirements.txt   # Dépendances Python
├── Dockerfile        # Configuration Docker
├── README.md         # Documentation
└── rapport/          # Rapport LaTeX
    ├── rapport.tex
    └── rapport.pdf
```

## Installation et utilisation

### 1. Environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Entraînement du modèle
```bash
python train_model.py
```

### 3. Lancement de l'API
```bash
python app.py
```

### 4. Test de l'API
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": [0.0, 0.1, 0.2, ...]}'  # 784 valeurs
```

### 5. Docker
```bash
# Construction de l'image
docker build -t mnist-api .

# Lancement du conteneur
docker run -p 5000:5000 mnist-api
```

## MLflow
Pour visualiser les expérimentations :
```bash
mlflow ui
```

## Auteur
Étudiant ENSPY - Département Génie Informatique