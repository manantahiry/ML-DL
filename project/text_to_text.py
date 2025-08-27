"""
Team:
    - Douglas       3396
    - Sarobidy      3435
    - Manantahiry   3405
    - Robert        3415
    - Rova Karl
Traduction Text-to-Text : Malgache Officiel ➜ Dialecte Sud-Est
Étapes : chargement du dataset, nettoyage approfondi, manipulation, traduction via Transformer.
"""

import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from typing import List
import re

# ================================
#  Paramètres et modèle
# ================================
MODEL_NAME = "Dapor/Dapson"

print(" Chargement du modèle Transformer...")
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)
print(" Modèle chargé avec succès.")

# ================================
#  Fonctions de nettoyage & normalisation
# ================================

def clean_text(text: str) -> str:
    """
    Nettoyage de base :
    - suppression espaces multiples
    - suppression ponctuation inutile
    - passage en minuscules
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()                        # Passage en minuscules + suppression espaces
    text = re.sub(r'[^\w\s\'-]', '', text)             # Supprime ponctuation sauf tiret/apostrophe
    text = re.sub(r'\s+', ' ', text)                   # Réduction des espaces multiples
    return text

def is_valid(text: str) -> bool:
    """
    Vérifie si un texte est exploitable (ni trop court, ni vide).
    """
    return isinstance(text, str) and 4 < len(text.strip()) < 200

# ================================
#  Chargement et prétraitement du dataset
# ================================
JSON_PATH = "data/text/dialect.json"
print(f"Lecture du fichier : {JSON_PATH}")
df = pd.read_json(JSON_PATH)

# Vérification des colonnes obligatoires
expected_columns = {"phrase_officiel", "phrase_dialecte"}
if not expected_columns.issubset(df.columns):
    raise ValueError(f"Le fichier CSV doit contenir les colonnes : {expected_columns}")

# Nettoyage des colonnes
df["officiel"] = df["officiel"].astype(str).apply(clean_text)
df["dialecte"] = df["dialecte"].astype(str).apply(clean_text)

# Suppression des lignes vides ou invalides
df = df[df["officiel"].apply(is_valid)]
df = df.drop_duplicates(subset=["officiel"])

print(f"Dataset nettoyé : {len(df)} phrases valides")

# ================================
#  Traduction par Transformer
# ================================

def translate_text(text: str) -> str:
    """
    Traduit une phrase du malgache officiel vers un dialecte.
    """
    if not text:
        return ""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    output = model.generate(**tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Boucle de traduction avec affichage
translations: List[str] = []

print("\n Traduction des phrases en cours...\n")
for i, row in df.iterrows():
    source = row["officiel"]
    reference = row["dialecte"]
    predicted = translate_text(source)
    translations.append(predicted)
    print(f"{i+1}.  Source   : {source}")
    print(f"     Cible   : {reference}")
    print(f"     Modèle  : {predicted}\n")

# Ajout des résultats dans le DataFrame
df["traduction_modele"] = translations

# ================================
#  Sauvegarde des résultats
# ================================
OUTPUT_JSON = "data/text/traductions_resultats.json"
df.to_json(OUTPUT_JSON)
print(f"\n Traductions sauvegardées dans : {OUTPUT_JSON}")