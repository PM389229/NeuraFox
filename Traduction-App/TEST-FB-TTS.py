import torch
from transformers import VitsModel, AutoTokenizer
from scipy.io.wavfile import write as write_wav
import os
import warnings

# Supprimer les avertissements pour garder la console propre
warnings.filterwarnings("ignore", category=UserWarning)

# Dictionnaire des langues et des phrases de test
# Ajoutez ou modifiez les langues que vous souhaitez tester
languages_to_test = {
    "anglais": {
        "model_id": "facebook/mms-tts-eng",
        "text": "Hello, this is a test of the multilingual text-to-speech model from Facebook."
    },
    "allemand": {
        "model_id": "facebook/mms-tts-deu",
        "text": "Hallo, dies ist ein Test des mehrsprachigen Text-to-Speech-Modells von Facebook."
    },
    "espagnol": {
        "model_id": "facebook/mms-tts-spa",
        "text": "Hola, esto es una prueba del modelo de texto a voz multilingüe de Facebook."
    },
    "ouïghour": {
        "model_id": "facebook/mms-tts-uig-script_arabic",
        "text": "تەكىست-تو-سوپىچ مودېللىرىنى سىناش"
    },
    "russe": {
        "model_id": "facebook/mms-tts-rus",
        "text": "Привет, это тест многоязычной модели преобразования текста в речь от Facebook."
    }
}

# Définir le périphérique (GPU si disponible, sinon CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation du périphérique: {device}")

# Créer un dossier pour les fichiers audio de sortie
if not os.path.exists("mms_audio_tests"):
    os.makedirs("mms_audio_tests")

def generate_audio(model_id, text, output_filename):
    """
    Charge un modèle MMS et génère un fichier audio.
    """
    try:
        print(f"\n--- Test du modèle : {model_id} ---")
        print(f"Texte d'entrée : {text}")

        # Chargement du modèle et du tokenizer
        model = VitsModel.from_pretrained(model_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Tokenization et génération de la voix
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs).waveform

        # Écriture du fichier WAV
        sampling_rate = model.config.sampling_rate
        output_path = os.path.join("mms_audio_tests", output_filename)
        write_wav(output_path, sampling_rate, output.squeeze().cpu().numpy())
        
        print(f"Fichier audio généré avec succès : {output_path}")

    except Exception as e:
        print(f"Erreur lors de la génération pour {model_id}: {e}")

# Exécuter les tests pour chaque langue
for lang, data in languages_to_test.items():
    model_id = data["model_id"]
    text = data["text"]
    # Nettoyer le nom de fichier pour éviter les caractères spéciaux
    output_filename = f"{lang.replace(' ', '_')}.wav"
    generate_audio(model_id, text, output_filename)

print("\n--- Tests terminés ---")
print("Vérifiez le dossier 'mms_audio_tests' pour écouter les résultats.")
