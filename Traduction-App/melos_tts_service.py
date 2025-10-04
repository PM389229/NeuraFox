# melo_tts_service.py
# Script unifié pour la synthèse vocale Japonais (JA) et Mandarin (ZH) utilisant MeloTTS
# Exécuté via subprocess par translator_service.py

import sys
import json
import torch
import warnings
from melo.api import TTS as MeloTTS

# Supprimer le warning de PyTorch pour une sortie console plus propre
warnings.filterwarnings("ignore", category=UserWarning)

def generate_tts(text_to_speak, output_file_path, lang_code):
    """
    Génère un fichier audio WAV à partir du texte pour la langue spécifiée (JA ou ZH).
    """
    try:
        # Configuration de l'appareil
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Logique spécifique pour le Mandarin (ZH) pour éviter les erreurs sur certaines plateformes
        if lang_code == 'ZH':
            # Force la désactivation de MPS (Apple Silicon) si elle était présente, comme dans le script original
            torch.backends.mps.is_available = lambda: False
            # On pourrait aussi définir l'environnement ici si nécessaire, 
            # mais cela est géré dans le subprocess de translator_service.py

        # Initialisation du modèle MeloTTS
        # MeloTTS prend 'ZH' ou 'JP'
        model = MeloTTS(language=lang_code, device=device)
        model.speaker_id = model.hps.data.spk2id[lang_code]

        # Limite de longueur du texte (pour éviter les erreurs du modèle MeloTTS sur les longs textes)
        if len(text_to_speak) > 200:
            text_to_speak = text_to_speak[:200]

        # Génération du fichier audio
        # Note: MeloTTS crée par défaut des fichiers WAV
        model.tts_to_file(text_to_speak, model.speaker_id, output_file_path, speed=1.0)
        return True
    except Exception as e:
        # Sortie d'erreur vers stderr, capturée par subprocess
        print(f"Erreur lors de la génération TTS ({lang_code}) : {e}", file=sys.stderr)
        return False

if __name__ == '__main__':
    # Le script attend un argument JSON contenant le texte, le chemin de sortie et la langue
    try:
        data = json.loads(sys.argv[1])
        text = data.get('text', '')
        output_path = data.get('output_path', '')
        lang_code = data.get('lang', '').upper() # Attendu: 'JA' ou 'ZH'

        if not text or not output_path or lang_code not in ['JA', 'ZH']:
            raise ValueError("Arguments d'entrée manquants ou invalides.")

        if generate_tts(text, output_path, lang_code):
            print("Success")
        else:
            print("Failure")
            sys.exit(1)
            
    except Exception as e:
        print(f"Erreur lors de l'exécution du script d'entrée/sortie : {e}", file=sys.stderr)
        sys.exit(1)
