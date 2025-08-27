"""
Team:
    - Douglas       3396
    - Sarobidy      3435
    - Manantahiry   3405
    - Robert        3415
    - Rova Karl

Voice-to-Voice Malgache : Audio officiel -> texte transcrit -> texte dialectal -> audio dialectal

Pipeline complet avec :
- Prétraitement audio (wav 16kHz mono)
- Transcription Whisper
- Traduction Transformer (local)
- Synthèse MMS-TTS-MLG
- Interface Gradio

Prérequis :
- Un modèle Transformer local pour la traduction (ex: MarianMTModel)
- Fichiers audio dans un dossier
"""

import os
import uuid
from glob import glob
from pydub import AudioSegment, effects
import whisper
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
import gradio as gr

# --------------------------
# Configuration chemins
# --------------------------

RAW_AUDIO_DIR = "data/audio/"             # dossier fichiers audio originaux
CLEAN_AUDIO_DIR = "data/audios/clean_wav"   # dossier fichiers wav nettoyés
OUTPUT_DIR = "data/audios/voice_to_voice"   # dossier sortie audio dialectal

os.makedirs(CLEAN_AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# Prétraitement audio
# --------------------------

def preprocess_audio(input_path: str, output_dir: str = CLEAN_AUDIO_DIR) -> str:
    """
    Convertit un fichier audio en wav mono 16kHz et normalise.
    """
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio = effects.normalize(audio)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.wav")
        audio.export(output_path, format="wav")
        print(f" Audio preprocessé : {output_path}")
        return output_path
    except Exception as e:
        print(f" Erreur prétraitement audio {input_path}: {e}")
        return None

def preprocess_all_audios(raw_dir: str) -> list:
    """
    Prétraite tous les audios du dossier raw_dir.
    Retourne la liste des fichiers wav nettoyés.
    """
    audio_files = glob(os.path.join(raw_dir, "*.*"))
    cleaned_files = []
    for f in audio_files:
        cleaned = preprocess_audio(f)
        if cleaned:
            cleaned_files.append(cleaned)
    return cleaned_files

# --------------------------
# Chargement modèles ML
# --------------------------

print(" Chargement Whisper (voice-to-text)...")
whisper_model = whisper.load_model("small")
print(" Whisper chargé.")

print(" Chargement modèle Transformer pour traduction texte...")
# Remplace ce chemin par le chemin local de ton modèle Transformer
transformer_model_name = "speechbrain/sepformer-wsj02mix"
tokenizer = MarianTokenizer.from_pretrained(transformer_model_name)
transformer_model = MarianMTModel.from_pretrained(transformer_model_name)
print(" Transformer chargé.")

print(" Chargement MMS-TTS-MLG (text-to-speech)...")
tts_model = TTS(model_name="facebook/mms-tts-mlg")
print(" TTS chargé.")

# --------------------------
# Pipeline Voice-to-Voice
# --------------------------

def voice_to_voice_pipeline(audio_path: str):
    """
    1. Transcription audio officiel -> texte
    2. Traduction texte officiel -> texte dialectal
    3. Synthèse voix dialectale
    Retourne : texte source, texte traduit, chemin audio dialectal généré
    """
    if not audio_path:
        return "Aucun fichier audio fourni", None, None

    # Étape 1 : transcription Whisper
    transcription_result = whisper_model.transcribe(audio_path, language="fr")
    source_text = transcription_result.get("text", "").strip()

    if not source_text:
        return "Transcription échouée ou vide", None, None

    # Étape 2 : traduction Transformer
    inputs = tokenizer(source_text, return_tensors="pt", padding=True, truncation=True)
    outputs = transformer_model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Étape 3 : synthèse vocale dialectale
    output_filename = f"voice_to_voice_{uuid.uuid4().hex[:8]}.wav"
    output_path = os