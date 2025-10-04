# config.py
import os

# Chemins
base_path = os.path.dirname(os.path.abspath(__file__))
MODEL_LOCAL_DIR = os.path.join(base_path, "hf-seamless-m4t-medium")
CACHE_DIR = os.path.join(base_path, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Paramètres Audio
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01 # Seuil RMS pour considérer l'audio comme du silence
SILENCE_DURATION = 2.0   # Durée de silence requise pour arrêter l'enregistrement
MAX_RECORDING_TIME = 60.0 # Durée maximale d'enregistrement

# Données des Langues
LANGUAGES_DATA = {
    "Français": {"seamless_code": "__fra__", "vosk_model_id": "vosk-model-small-fr-0.22", "tts_lang_code": "FR"},
    "Anglais": {"seamless_code": "__eng__", "vosk_model_id": "vosk-model-small-en-us-0.22", "tts_lang_code": "EN"},
    "Japonais": {"seamless_code": "__jpn__", "vosk_model_id": None, "tts_lang_code": "JA"},
    "Allemand": {"seamless_code": "__deu__", "vosk_model_id": "vosk-model-small-de-0.21", "tts_lang_code": "DE"},
    "Espagnol": {"seamless_code": "__spa__", "vosk_model_id": "vosk-model-small-es-0.42", "tts_lang_code": "ES"},
    "Russe": {"seamless_code": "__rus__", "vosk_model_id": "vosk-model-small-ru-0.10", "tts_lang_code": "RU"},
    "Portugais": {"seamless_code": "__por__", "vosk_model_id": "vosk-model-small-pt-0.3", "tts_lang_code": "PT"},
    "Mandarin": {"seamless_code": "__cmn__", "vosk_model_id": "vosk-model-small-cn-0.22", "tts_lang_code": "ZH"},
    "Arabe": {"seamless_code": "__arb__", "vosk_model_id": "vosk-model-ar-mgb2-0.4", "tts_lang_code": "AR"},
}

# Modèles MMS-TTS
MMS_TTS_MODELS = {
    "FR": "facebook/mms-tts-fra",
    "EN": "facebook/mms-tts-eng",
    "DE": "facebook/mms-tts-deu",
    "ES": "facebook/mms-tts-spa",
    "RU": "facebook/mms-tts-rus",
    "PT": "facebook/mms-tts-por",
    "AR": "facebook/mms-tts-ara",
}

# URLs Vosk
VOSK_URLS = {
    "Français": "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
    "Anglais": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "Allemand": "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip",
    "Espagnol": "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip",
    "Russe": "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",
    "Portugais": "https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip",
    "Mandarin": "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip",
    "Arabe": "https://alphacephei.com/vosk/models/vosk-model-ar-mgb2-0.4.zip",
}
