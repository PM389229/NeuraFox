# model_manager.py
import os
import streamlit as st
import logging
import requests, zipfile, io
import shutil
import torch
import numpy as np
import soundfile as sf
import tempfile
import wave
import queue
import time

from vosk import Model, KaldiRecognizer
from transformers import AutoProcessor, SeamlessM4TModel, VitsModel, AutoTokenizer
try:
    import whisper 
except ImportError:
    # Whisper n'est pas une dépendance obligatoire mais est utilisé en fallback
    # La librairie est "openai-whisper"
    whisper = None

# Importation des constantes du fichier config.py
# NOTE: Le fichier config.py doit être présent pour que ce fichier fonctionne
from config import (
    MODEL_LOCAL_DIR, LANGUAGES_DATA, MMS_TTS_MODELS, VOSK_URLS, CACHE_DIR,
    SAMPLE_RATE, SILENCE_THRESHOLD, SILENCE_DURATION, MAX_RECORDING_TIME
) 

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='LOG: %(message)s')

class ModelManager:
    def __init__(self):
        # Les modèles Vosk chargés en mémoire (Model objects)
        self.vosk_loaded_models = {}

    # --- SeamlessM4T (Traduction) ---
    @st.cache_resource
    def get_translator_pipeline(_self): 
        """
        Charge et met en cache le modèle et le processeur SeamlessM4T.
        Retourne (model, processor).
        """
        
        local_path = MODEL_LOCAL_DIR
        # Utilisation de la version 'medium' comme convenu
        hf_model_name = "facebook/seamless-m4t-medium"
        model_source = local_path
        local_only = True 
        
        if os.path.exists(local_path) and os.path.isdir(local_path):
            logging.info(f"SeamlessM4T trouvé localement : {local_path}.")
            model_source = local_path
        else:
            logging.info(f"SeamlessM4T non trouvé localement. Tentative de chargement via Hugging Face.")
            st.warning("⚠️ Le modèle SeamlessM4T est très volumineux. Un téléchargement est en cours.")
            model_source = hf_model_name
            local_only = False # Autorise le téléchargement si nécessaire
        
        logging.info("Chargement du Modèle et Processeur SeamlessM4T...")
        try:
            # S'assurer d'utiliser la bonne classe SeamlessM4TModel
            processor = AutoProcessor.from_pretrained(model_source, local_files_only=local_only)
            model = SeamlessM4TModel.from_pretrained(model_source, local_files_only=local_only)
            
            logging.info("Modèle SeamlessM4T chargé avec succès.")
            return model, processor 
            
        except Exception as e:
            logging.error(f"Erreur lors du chargement de SeamlessM4T: {e}")
            st.error(f"Erreur: Le modèle SeamlessM4T n'a pas pu être chargé. Détail: {e}")
            return None, None 

    # --- Gestion du TTS (MMS/Melo) ---
    def download_tts_model(self, lang_code):
        """
        Télécharge ou prépare le modèle TTS pour la langue donnée.
        Retourne le chemin du modèle.
        """
        if lang_code in ["JA", "ZH"]:
            # On ne fait que préparer le cache pour MeloTTS (pas de téléchargement direct ici)
            model_path = os.path.join(CACHE_DIR, f"melo_{lang_code.lower()}")
            os.makedirs(model_path, exist_ok=True)
            if not os.path.exists(os.path.join(model_path, "placeholder.txt")):
                 with open(os.path.join(model_path, "placeholder.txt"), "w") as f:
                     f.write(f"MeloTTS {lang_code} placeholder.")
            return model_path

        # Pour les autres langues (MMS-TTS VITS)
        if lang_code not in MMS_TTS_MODELS:
            return None

        model_name = MMS_TTS_MODELS[lang_code]
        model_path = os.path.join(CACHE_DIR, f"mms_{lang_code}")

        if not os.path.exists(model_path) or not os.listdir(model_path):
            st.info(f"Téléchargement du modèle MMS-TTS pour {lang_code}...")
            # Précharger le tokenizer et le modèle VITS
            try:
                AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
                VitsModel.from_pretrained(model_name, cache_dir=model_path)
                logging.info(f"MMS-TTS {lang_code} téléchargé dans {model_path}")
            except Exception as e:
                st.error(f"Erreur lors du téléchargement de MMS-TTS {lang_code}: {e}")
                logging.error(f"Erreur MMS TTS: {e}")
                return None
        return model_path

    # --- Gestion du VOSK (ASR) ---
    def download_vosk_model(self, lang_name):
        """
        Télécharge et charge le modèle Vosk (ASR).
        Retourne l'objet Model Vosk ou None.
        """
        vosk_id = LANGUAGES_DATA[lang_name]["vosk_model_id"]
        if vosk_id is None:
            return None

        # 1. Tenter de charger depuis le cache en mémoire
        if lang_name in self.vosk_loaded_models:
             return self.vosk_loaded_models[lang_name]

        # 2. Tenter de charger depuis le dossier local
        path = os.path.join(CACHE_DIR, vosk_id)
        if os.path.exists(path) and os.listdir(path):
            try:
                model = Model(path)
                self.vosk_loaded_models[lang_name] = model
                logging.info(f"Modèle Vosk pour {lang_name} chargé depuis le cache local.")
                return model
            except Exception as e:
                logging.error(f"Échec du chargement du modèle Vosk local existant: {e}")
        
        # 3. Téléchargement et extraction
        url = VOSK_URLS.get(lang_name)
        if url is None:
            return None

        st.info(f"Téléchargement du modèle Vosk pour {lang_name}...")
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                # Extraire dans le CACHE_DIR
                zf.extractall(CACHE_DIR)
                
                # Le dossier extrait doit correspondre au vosk_id
                extracted_dir_name = zf.namelist()[0].split('/')[0]
                extracted_dir = os.path.join(CACHE_DIR, extracted_dir_name)
                
                # S'assurer que le dossier porte le nom de l'ID pour le chargement
                if extracted_dir_name != vosk_id and os.path.exists(extracted_dir):
                    final_path = os.path.join(CACHE_DIR, vosk_id)
                    if os.path.exists(final_path):
                         shutil.rmtree(final_path) # Supprime l'ancien pour éviter les conflits
                    shutil.move(extracted_dir, final_path)
                    
            model = Model(path) # Charge depuis le chemin renommé
            self.vosk_loaded_models[lang_name] = model
            st.success(f"Modèle Vosk {lang_name} téléchargé et chargé.")
            return model
            
        except Exception as e:
            st.error(f"Erreur lors du téléchargement/chargement de Vosk : {e}")
            logging.error(f"Erreur Vosk: {e}")
            return None

    # --- GESTION DU WHISPER ---
    @st.cache_resource
    def get_whisper_model(_self, model_size: str = "small"):
        """
        Charge et met en cache le modèle Whisper (utilisé comme fallback ou pour JA/ZH).
        """
        if whisper is None:
            st.error("❌ La librairie Whisper n'est pas installée. Impossible de transcrire JA/ZH ou d'utiliser le fallback.")
            return None
            
        logging.info(f"Chargement du modèle Whisper '{model_size}'...")
        try:
            # Assurez-vous d'utiliser un appareil disponible ou forcez le CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisper.load_model(model_size, device=device) 
            logging.info(f"Modèle Whisper '{model_size}' chargé avec succès sur {device}.")
            return model
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du modèle Whisper '{model_size}' : {e}")
            return None