# -*- coding: utf-8 -*-

import sys
import json
import torch
import warnings
from melo.api import TTS as MeloTTS

# Supprimer le warning de PyTorch pour plus de clarté
warnings.filterwarnings("ignore", category=UserWarning)

def generate_tts(text_to_speak, output_file_path):
    """
    Génère un fichier audio MP3 à partir du texte anglais fourni en utilisant MeloTTS.
    """
    try:
        # Utilise le GPU si disponible, sinon le CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Le modèle TTS anglais est un modèle MeloTTS, chargé par le nom de la langue
        model = MeloTTS(language='EN', device=device)
        model.speaker_id = model.hps.data.spk2id['EN']
        
        # Le modèle MeloTTS a des limites sur la longueur du texte.
        # On le coupe pour éviter les erreurs.
        if len(text_to_speak) > 200:
            text_to_speak = text_to_speak[:200]
        
        model.tts_to_file(text_to_speak, model.speaker_id, output_file_path, speed=1.0)
        return True
    except Exception as e:
        print(f"Erreur lors de la génération TTS : {e}", file=sys.stderr)
        return False

if __name__ == '__main__':
    # Récupérer les données passées en ligne de commande
    try:
        data = json.loads(sys.argv[1])
        text = data['text']
        output_path = data['output_path']
        
        if generate_tts(text, output_path):
            # En cas de succès, on renvoie "Success" sur la sortie standard
            print("Success")
        else:
            # En cas d'échec, on renvoie "Failure"
            print("Failure")
            sys.exit(1)
            
    except Exception as e:
        print(f"Erreur lors de l'exécution du script : {e}", file=sys.stderr)
        sys.exit(1)
