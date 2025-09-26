import os
import requests
import zipfile
import io
import shutil

# Dictionnaire des modèles à télécharger et leurs URLs.
# Inclut les versions "small" et "lightweight" pour toutes les langues disponibles.
MODELS_TO_DOWNLOAD = {
    "vosk-model-small-en-us-0.15": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "vosk-model-small-en-us-zamia-0.5": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-zamia-0.5.zip",
    "vosk-model-small-en-in-0.4": "https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip",
    "vosk-model-small-cn-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip",
    "vosk-model-small-ru-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",
    "vosk-model-small-fr-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
    "vosk-model-small-fr-pguyot-0.3": "https://alphacephei.com/vosk/models/vosk-model-small-fr-pguyot-0.3.zip",
    "vosk-model-small-de-zamia-0.3": "https://alphacephei.com/vosk/models/vosk-model-small-de-zamia-0.3.zip",
    "vosk-model-small-de-0.15": "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip",
    "vosk-model-small-es-0.42": "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip",
    "vosk-model-small-pt-0.3": "https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip",
    "vosk-model-small-tr-0.3": "https://alphacephei.com/vosk/models/vosk-model-small-tr-0.3.zip",
    "vosk-model-small-vn-0.4": "https://alphacephei.com/vosk/models/vosk-model-small-vn-0.4.zip",
    "vosk-model-small-it-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-it-0.22.zip",
    "vosk-model-small-nl-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-nl-0.22.zip",
    "vosk-model-small-ca-0.4": "https://alphacephei.com/vosk/models/vosk-model-small-ca-0.4.zip",
    "vosk-model-small-ar-tn-0.1-linto": "https://alphacephei.com/vosk/models/vosk-model-small-ar-tn-0.1-linto.zip",
    "vosk-model-small-fa-0.42": "https://alphacephei.com/vosk/models/vosk-model-small-fa-0.42.zip",
    "vosk-model-small-fa-0.5": "https://alphacephei.com/vosk/models/vosk-model-small-fa-0.5.zip",
    "vosk-model-small-uk-v3-nano": "https://alphacephei.com/vosk/models/vosk-model-small-uk-v3-nano.zip",
    "vosk-model-small-uk-v3-small": "https://alphacephei.com/vosk/models/vosk-model-small-uk-v3-small.zip",
    "vosk-model-small-kz-0.15": "https://alphacephei.com/vosk/models/vosk-model-small-kz-0.15.zip",
    "vosk-model-small-sv-rhasspy-0.15": "https://alphacephei.com/vosk/models/vosk-model-small-sv-rhasspy-0.15.zip",
    "vosk-model-small-ja-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip",
    "vosk-model-small-eo-0.42": "https://alphacephei.com/vosk/models/vosk-model-small-eo-0.42.zip",
    "vosk-model-small-hi-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip",
    "vosk-model-small-cs-0.4-rhasspy": "https://alphacephei.com/vosk/models/vosk-model-small-cs-0.4-rhasspy.zip",
    "vosk-model-small-pl-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-pl-0.22.zip",
    "vosk-model-small-uz-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-uz-0.22.zip",
    "vosk-model-small-ko-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-ko-0.22.zip",
    "vosk-model-small-gu-0.42": "https://alphacephei.com/vosk/models/vosk-model-small-gu-0.42.zip",
    "vosk-model-small-tg-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-tg-0.22.zip",
    "vosk-model-small-te-0.42": "https://alphacephei.com/vosk/models/vosk-model-small-te-0.42.zip",
    "vosk-model-br-0.8": "https://alphacephei.com/vosk/models/vosk-model-br-0.8.zip",
}

def download_and_extract_model(model_name):
    """
    Télécharge et extrait un modèle Vosk spécifique.
    Retourne True en cas de succès, False sinon.
    """
    if model_name not in MODELS_TO_DOWNLOAD:
        print(f"Erreur: Le modèle '{model_name}' n'est pas dans la liste de téléchargement.")
        return False

    url = MODELS_TO_DOWNLOAD[model_name]
    target_dir = os.path.join(os.getcwd(), model_name)

    if os.path.exists(target_dir):
        print(f"Le modèle '{model_name}' existe déjà. Téléchargement non nécessaire.")
        return True

    try:
        print(f"Téléchargement du modèle '{model_name}' depuis {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_root_dir = zip_ref.namelist()[0].split('/')[0]
            zip_ref.extractall(os.getcwd())

        if zip_root_dir != model_name:
            shutil.move(zip_root_dir, model_name)
        
        print(f"Le modèle '{model_name}' a été téléchargé et extrait avec succès.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Erreur de téléchargement pour '{model_name}': {e}")
        return False
    except zipfile.BadZipFile:
        print(f"Erreur: Le fichier téléchargé pour '{model_name}' n'est pas un fichier zip valide.")
        return False
    except Exception as e:
        print(f"Une erreur inattendue est survenue lors du téléchargement: {e}")
        return False
