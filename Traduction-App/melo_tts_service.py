# melo_tts_service.py
# Script unifié pour la synthèse vocale Japonais (JP) et Mandarin (ZH) utilisant MeloTTS
# Exécuté via subprocess par translator_service.py

import sys
import json
import torch
import warnings
import MeCab
import os 
import unidic 


# --- PATCH UNICODE WINDOWS ---
# Empêche les prints internes de MeloTTS qui provoquent des UnicodeEncodeError sur Windows
import sys
class SafeWriter:
    def write(self, s):
        try:
            sys.__stdout__.write(s.encode("utf-8", errors="ignore").decode("utf-8"))
        except:
            pass
    def flush(self):
        pass

sys.stdout = SafeWriter()
sys.stderr = SafeWriter()
# --- FIN PATCH ---



# --- FIX PRÉ-INITIALISATION MECAB - VERSION COMPATIBLE WINDOWS / UNIDIC 3.1.0 ---
try:
    import MeCab

    # Chemin absolu vers UniDic
    MECAB_DIC_PATH = r"C:\Users\pmgue\Neurafox\unidic-mecab-2.1.2"

    # Normalisation des chemins pour Windows
    dic_path = os.path.normpath(MECAB_DIC_PATH)

    # Sur Windows, -r dicrc n'est pas nécessaire et provoque des erreurs
    TAGGER_ARGS = f'-d "{dic_path}"'

    # Test d'initialisation
    MeCab.Tagger(TAGGER_ARGS)
    print(f"LOG: MeCab initialisé avec succès. Args: {TAGGER_ARGS}", file=sys.stderr)

except Exception as e:
    print(f"ALERTE FATALE LOG: La pré-initialisation MeCab a échoué. Cause: {e}", file=sys.stderr)
    sys.exit(1)



# ------------------------------------------------

# L'importation de MeloTTS est la première chose qui pourrait échouer
try:
    from melo.api import TTS as MeloTTS
    print("LOG: Importation de melo.api.TTS réussie.", file=sys.stderr)
except ImportError as e:
    print(f"FATAL LOG: Échec de l'importation de melo.api.TTS: {e}", file=sys.stderr)
    sys.exit(1) # Sortie immédiate si l'importation échoue.

# Supprimer le warning de PyTorch pour une sortie console plus propre
warnings.filterwarnings("ignore", category=UserWarning)





def generate_tts(text_to_speak, output_file_path, lang_code):
    """
    Génère un fichier audio WAV à partir du texte pour la langue spécifiée (JA ou ZH).
    """
    print(f"LOG: Démarrage de la génération TTS pour la langue: {lang_code}", file=sys.stderr)
    try:
        # Configuration de l'appareil
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"LOG: Appareil de traitement PyTorch détecté: {device}", file=sys.stderr)

        # Logique spécifique pour le Mandarin (ZH)
        if lang_code == 'ZH':
            print("LOG: Logique spécifique ZH activée (désactivation potentielle de MPS).", file=sys.stderr)
            torch.backends.mps.is_available = lambda: False

        # Initialisation du modèle MeloTTS
        print(f"LOG: Initialisation du modèle MeloTTS pour '{lang_code}'...", file=sys.stderr)
        model = MeloTTS(language=lang_code, device=device)
        print("LOG: Modèle MeloTTS créé, vérification des speakers...", file=sys.stderr)
        print(f"LOG: Clés spk2id disponibles : {list(model.hps.data.spk2id.keys())}", file=sys.stderr)
        if lang_code not in model.hps.data.spk2id:
            raise KeyError(f"Langue '{lang_code}' introuvable dans spk2id !")
        model.speaker_id = model.hps.data.spk2id[lang_code]
        print(f"LOG: Speaker_id pour '{lang_code}' = {model.speaker_id}", file=sys.stderr)
        print("LOG: Modèle MeloTTS initialisé avec succès.", file=sys.stderr)

        # Limite de longueur du texte
        original_length = len(text_to_speak)
        if original_length > 200:
            text_to_speak = text_to_speak[:200]
            print(f"LOG: Texte tronqué de {original_length} à 200 caractères.", file=sys.stderr)
        print(f"LOG: Texte final envoyé à MeloTTS : '{text_to_speak[:50]}{'...' if len(text_to_speak)>50 else ''}'", file=sys.stderr)

        # Avant génération audio
        print(f"LOG: Chemin de sortie du WAV : {output_file_path}", file=sys.stderr)
        print("LOG: Début de model.tts_to_file...", file=sys.stderr)

        # Génération du fichier audio
        model.tts_to_file(text_to_speak, model.speaker_id, output_file_path, speed=1.0)

        print("LOG: model.tts_to_file terminé avec succès.", file=sys.stderr)
        print("LOG: Fichier audio généré avec succès.", file=sys.stderr)
        return True

    except Exception as e:
        # Sortie d'erreur vers stderr, capturée par subprocess
        import traceback
        print(f"Erreur lors de la génération TTS ({lang_code}) : {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False







if __name__ == '__main__':
    # Log de l'environnement Python utilisé
    print(f"LOG: Script melo_tts_service.py démarré.", file=sys.stderr)
    print(f"LOG: Exécutable Python utilisé: {sys.executable}", file=sys.stderr)

    # Le script attend un argument JSON
    try:
        if len(sys.argv) < 2:
            raise ValueError("Argument JSON manquant. Le script nécessite un argument.")
            
        data = json.loads(sys.argv[1])
        text = data.get('text', '')
        output_path = data.get('output_path', '')
        lang_code = data.get('lang', '').upper()
        
        # Log des arguments reçus
        print(f"LOG: Args reçus - Texte (début): '{text[:50]}...', Path: '{output_path}', Lang: '{lang_code}'", file=sys.stderr)

        if not text or not output_path or lang_code not in ['JP', 'ZH']:
            raise ValueError("Arguments d'entrée manquants ou invalides.")

        if generate_tts(text, output_path, lang_code):
            print("Success")
        else:
            print("Failure")
            sys.exit(1)
            
    except Exception as e:
        print(f"Erreur lors de l'exécution du script d'entrée/sortie : {e}", file=sys.stderr)
        sys.exit(1)
