"""
Team:
    - Douglas       3396
    - Sarobidy      3435
    - Manantahiry   3405
    - Robert        3415
    - Rova Karl

Partie 2 – Voice-to-Text & Text-to-Voice avec dataset audio en dossier

- Prétraitement audio : conversion wav 16k mono + normalisation
- Chargement dynamique des fichiers wav nettoyés
- Interface Gradio pour transcription et synthèse vocale
"""

import os
import uuid
from glob import glob
from pydub import AudioSegment, effects
import whisper
from TTS.api import TTS
import gradio as gr

# -----------------------------------
# Configurations dossier
# -----------------------------------

RAW_AUDIO_DIR = "data/audio/"          # Dossier fichiers audio bruts (mp3, wav, m4a, etc)
CLEAN_AUDIO_DIR = "data/audios/clean_wav" # Dossier fichiers wav nettoyés
OUTPUT_DIR = "data/audios/voice_text_outputs" # Dossier sortie synthèse vocale

os.makedirs(CLEAN_AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------
# Fonction prétraitement audio
# -----------------------------------

def preprocess_audio_file(input_path: str, output_dir: str = CLEAN_AUDIO_DIR) -> str:
    """
    Convertit un fichier audio en WAV mono 16kHz et normalise le volume.
    """
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio = effects.normalize(audio)

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.wav")
        audio.export(output_path, format="wav")
        print(f"Converti : {input_path} → {output_path}")
        return output_path
    except Exception as e:
        print(f"Erreur conversion {input_path}: {e}")
        return None

# -----------------------------------
# Prétraitement de tout le dossier
# -----------------------------------

def preprocess_all_audios(raw_dir: str) -> list:
    """
    Convertit tous les fichiers audio du dossier raw_dir en WAV nettoyés.
    Retourne la liste des chemins nettoyés.
    """
    audio_paths = glob(os.path.join(raw_dir, "*.*"))
    cleaned_paths = []
    for path in audio_paths:
        cleaned = preprocess_audio_file(path)
        if cleaned:
            cleaned_paths.append(cleaned)
    return cleaned_paths

# -----------------------------------
# Chargement modèles ML
# -----------------------------------

print(" Chargement Whisper pour transcription...")
whisper_model = whisper.load_model("small")
print(" Whisper prêt.")

print(" Chargement MMS-TTS pour synthèse vocale...")
tts_model = TTS(model_name="facebook/mms-tts-mlg")
print(" TTS prêt.")

# -----------------------------------
# Fonctions Voice-to-Text et Text-to-Voice
# -----------------------------------

def voice_to_text(audio_path: str) -> str:
    if not audio_path:
        return "Aucun fichier audio fourni."
    result = whisper_model.transcribe(audio_path, language="fr")
    return result["text"].strip()

def text_to_voice(text: str) -> str:
    if not text.strip():
        return None
    filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
    output_path = os.path.join(OUTPUT_DIR, filename)
    tts_model.tts_to_file(text=text.strip(), file_path=output_path)
    return output_path

# -----------------------------------
# Interface Gradio
# -----------------------------------

def build_interface(cleaned_audio_files: list):
    with gr.Blocks(title=" Voice-to-Text & Text-to-Voice Malgache") as demo:
        gr.Markdown("#  Transcription & Synthèse vocale")

        with gr.Tab(" Voice-to-Text (Transcription)"):
            gr.Markdown("Sélectionne un fichier audio nettoyé pour transcrire.")
            audio_dropdown = gr.Dropdown(cleaned_audio_files, label="Audio WAV nettoyé", interactive=True)
            transcript_output = gr.Textbox(label="Texte transcrit")
            transcribe_btn = gr.Button("Transcrire")
            transcribe_btn.click(voice_to_text, inputs=audio_dropdown, outputs=transcript_output)

        with gr.Tab(" Text-to-Voice (Synthèse vocale)"):
            gr.Markdown("Saisis un texte dialectal pour générer la voix.")
            text_input = gr.Textbox(label="Texte dialectal", placeholder="Ex: Misaotsy be amin'ny fanampiana")
            audio_output = gr.Audio(label="Voix synthétique")
            synth_btn = gr.Button("Générer audio")
            synth_btn.click(text_to_voice, inputs=text_input, outputs=audio_output)

    return demo

# -----------------------------------
# Main
# -----------------------------------

if __name__ == "__main__":
    print(f"Prétraitement des audios dans : {RAW_AUDIO_DIR}")
    cleaned_audio_files = preprocess_all_audios(RAW_AUDIO_DIR)
    print(f" {len(cleaned_audio_files)} fichiers audio nettoyés disponibles.")

    interface = build_interface(cleaned_audio_files)
    interface.launch()
