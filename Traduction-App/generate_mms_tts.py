# generate_mms_tts.py
import sys, json, os
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

MMS_TTS_MODELS = {
    "FR": "facebook/mms-tts-fr",
    "EN": "facebook/mms-tts-en",
    "JA": "facebook/mms-tts-ja",
    "DE": "facebook/mms-tts-de",
    "ES": "facebook/mms-tts-es",
    "RU": "facebook/mms-tts-ru"
}

def download_model(lang_code):
    model_name = MMS_TTS_MODELS.get(lang_code)
    if model_name is None:
        raise ValueError(f"Aucun modèle MMS-TTS pour le code {lang_code}")
    model_path = os.path.join(CACHE_DIR, f"mms_{lang_code}")
    if not os.path.exists(model_path):
        print(f"Téléchargement du modèle {model_name} pour {lang_code}...")
        AutoProcessor.from_pretrained(model_name, cache_dir=model_path)
        AutoModelForSpeechSeq2Seq.from_pretrained(model_name, cache_dir=model_path)
    return model_path

def synthesize(text, lang_code, output_path):
    model_path = download_model(lang_code)
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)

    inputs = processor(text, return_tensors="pt")
    with torch.no_grad():
        speech = model.generate_speech(**inputs)

    # Sauvegarde en WAV
    sf.write(output_path, speech.numpy(), 16000)
    print("Success")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_mms_tts.py '<json_input>'")
        sys.exit(1)

    data = json.loads(sys.argv[1])
    text = data.get("text", "")
    lang_code = data.get("language", "EN")
    output_path = data.get("output_path", "output.wav")

    synthesize(text, lang_code, output_path)
