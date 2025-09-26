# -*- coding: utf-8 -*-
import sys
import json
import torch
import warnings
from melo.api import TTS as MeloTTS

warnings.filterwarnings("ignore", category=UserWarning)

def generate_tts(text_to_speak, output_file_path):
    try:
        # Forcer CPU et désactiver MPS pour éviter les erreurs sur Mac
        device = "cuda" if torch.cuda.is_available() else "cpu"  # reste string
        torch.backends.mps.is_available = lambda: False  # désactive MPS
        model = MeloTTS(language='ZH', device=device)
        model.speaker_id = model.hps.data.spk2id['ZH']

        # Limite de longueur du texte
        if len(text_to_speak) > 200:
            text_to_speak = text_to_speak[:200]

        model.tts_to_file(text_to_speak, model.speaker_id, output_file_path, speed=1.0)
        return True
    except Exception as e:
        print(f"Erreur lors de la génération TTS (Mandarin) : {e}", file=sys.stderr)
        return False

if __name__ == '__main__':
    try:
        data = json.loads(sys.argv[1])
        text = data['text']
        output_path = data['output_path']

        if generate_tts(text, output_path):
            print("Success")
        else:
            print("Failure")
            sys.exit(1)
    except Exception as e:
        print(f"Erreur lors de l'exécution du script (Mandarin) : {e}", file=sys.stderr)
        sys.exit(1)
