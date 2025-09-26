import streamlit as st
import whisper
import re
import tempfile, os, torch, wave, sounddevice as sd
import json
import numpy as np
from vosk import Model, KaldiRecognizer
from transformers import AutoProcessor, SeamlessM4TModel, AutoModelForSpeechSeq2Seq,VitsModel, AutoTokenizer
import subprocess
import queue
import time
import soundfile as sf
import requests, zipfile, io

# =======================
# CONFIGURATION ET DONNÉES DES LANGUES
# =======================

LANGUAGES_DATA = {
    "Français": {"seamless_code": "__fra__", "vosk_model_id": "vosk-model-small-fr-0.22", "tts_lang_code": "FR"},
    "Anglais": {"seamless_code": "__eng__", "vosk_model_id": "vosk-model-small-en-us-0.22", "tts_lang_code": "EN"},
    "Japonais": {"seamless_code": "__jpn__", "vosk_model_id": None, "tts_lang_code": "JA"},
    "Allemand": {"seamless_code": "__deu__", "vosk_model_id": "vosk-model-small-de-0.21", "tts_lang_code": "DE"},
    "Espagnol": {"seamless_code": "__spa__", "vosk_model_id": "vosk-model-small-es-0.42", "tts_lang_code": "ES"},
    "Russe": {"seamless_code": "__rus__", "vosk_model_id": "vosk-model-small-ru-0.10", "tts_lang_code": "RU"},
    # Nouveaux ajouts
    
    "Portugais": {"seamless_code": "__por__", "vosk_model_id": "vosk-model-small-pt-0.3", "tts_lang_code": "PT"},
    "Mandarin": {"seamless_code": "__cmn__", "vosk_model_id": "vosk-model-small-cn-0.22", "tts_lang_code": "ZH"},
    "Arabe": {"seamless_code": "__arb__", "vosk_model_id": "vosk-model-ar-mgb2-0.4", "tts_lang_code": "AR"},
}

# Modèles MMS-TTS (chinois retiré car on utilise MeloTTS)
MMS_TTS_MODELS = {
    "FR": "facebook/mms-tts-fra",
    "EN": "facebook/mms-tts-eng",
    "DE": "facebook/mms-tts-deu",
    "ES": "facebook/mms-tts-spa",
    "RU": "facebook/mms-tts-rus",
    # Ajouts
    
    "PT": "facebook/mms-tts-por",
    "AR": "facebook/mms-tts-ara",
    # ZH retiré - on utilise MeloTTS
}

VOSK_URLS = {
    "Français": "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
    "Anglais": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "Allemand": "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip",
    "Espagnol": "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip",
    "Russe": "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip", 
    # Ajouts
    
    "Portugais": "https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip",
    "Mandarin": "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip",
    "Arabe": "https://alphacephei.com/vosk/models/vosk-model-ar-mgb2-0.4.zip",
}

base_path = os.path.dirname(os.path.abspath(__file__))
MODEL_LOCAL_DIR = os.path.join(base_path, "hf-seamless-m4t-medium")
CACHE_DIR = os.path.join(base_path, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 2.0
MAX_RECORDING_TIME = 60.0
SAMPLE_RATE = 16000

# =======================
# FONCTIONS DE CHARGEMENT DE MODÈLES
# =======================

@st.cache_resource
def get_translator_pipeline():
    print("LOG: Chargement du modèle SeamlessM4T...")
    if not os.path.exists(MODEL_LOCAL_DIR):
        st.error(f"Le modèle SeamlessM4T est introuvable dans {MODEL_LOCAL_DIR}")
        st.stop()
    processor = AutoProcessor.from_pretrained(MODEL_LOCAL_DIR, local_files_only=True)
    model = SeamlessM4TModel.from_pretrained(MODEL_LOCAL_DIR, local_files_only=True)
    print("LOG: Modèle SeamlessM4T chargé avec succès")
    return model, processor

@st.cache_resource
def load_whisper_model():
    print("LOG: Chargement du modèle Whisper...")
    model = whisper.load_model("small", device="cpu")
    print("LOG: Modèle Whisper chargé")
    return model

def download_vosk_model(lang_name):
    vosk_id = LANGUAGES_DATA[lang_name]["vosk_model_id"]
    if vosk_id is None:
        print(f"LOG: Pas de modèle Vosk pour {lang_name}")
        return None

    path = os.path.join(CACHE_DIR, vosk_id)
    if os.path.exists(path) and os.listdir(path):
        print(f"LOG: Modèle Vosk pour {lang_name} déjà présent dans {path}")
        try:
            return Model(path)
        except Exception as e:
            print(f"ERROR: Échec chargement modèle existant: {e}")

    url = VOSK_URLS.get(lang_name)
    if url is None:
        print(f"LOG: Pas d'URL pour {lang_name}")
        return None

    st.info(f"Téléchargement du modèle Vosk pour {lang_name}...")
    print(f"LOG: Téléchargement Vosk depuis {url}")
    r = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall(CACHE_DIR)

        # Détecter automatiquement le vrai dossier extrait
        subdirs = [name.split("/")[0] for name in zf.namelist() if "/" in name]
        if subdirs:
            path = os.path.join(CACHE_DIR, subdirs[0])

    print(f"LOG: Modèle Vosk {lang_name} extrait dans {path}")

    # Vérifier que le dossier contient bien des fichiers de modèle
    required_files = ["am", "conf", "graph"]
    if not all(os.path.exists(os.path.join(path, f)) for f in required_files):
        print(f"ERROR: Dossier Vosk {path} ne contient pas tous les fichiers nécessaires")
        return None

    try:
        model = Model(path)
        print(f"LOG: Modèle Vosk {lang_name} chargé avec succès")
        return model
    except Exception as e:
        print(f"ERROR: Impossible de créer le modèle Vosk: {e}")
        return None

def download_mms_model(lang_code):
    """
    Télécharge ou prépare le modèle TTS pour la langue donnée.
    
    - Pour "JA" (Japonais) et "ZH" (Chinois) → utilise MeloTTS via script externe
    - Pour les autres langues → MMS-TTS (VITS)
    """
    if lang_code in ["JA", "ZH"]:
        # Vérifier si le modèle est déjà dans le cache
        model_path = os.path.join(CACHE_DIR, f"melo_{lang_code.lower()}")
        if os.path.exists(model_path) and os.listdir(model_path):
            print(f"LOG: MeloTTS {lang_code} déjà présent")
            return model_path

        lang_name = "japonais" if lang_code == "JA" else "chinois"
        st.info(f"Téléchargement / préparation du modèle MeloTTS {lang_name}...")
        print(f"LOG: Préparation MeloTTS {lang_code}")
        os.makedirs(model_path, exist_ok=True)
        # On suppose que melo.api est déjà installé et prêt à l'emploi
        print(f"LOG: MeloTTS {lang_code} prêt dans {model_path}")
        return model_path

    # Pour les autres langues (MMS-TTS)
    if lang_code not in MMS_TTS_MODELS:
        st.error(f"Pas de modèle MMS-TTS défini pour {lang_code}")
        return None

    model_name = MMS_TTS_MODELS[lang_code]
    model_path = os.path.join(CACHE_DIR, f"mms_{lang_code}")

    if not os.path.exists(model_path) or not os.listdir(model_path):
        st.info(f"Téléchargement du modèle MMS-TTS pour {lang_code}...")
        print(f"LOG: Téléchargement MMS-TTS {lang_code} depuis {model_name}")
        # Précharger le tokenizer et le modèle VITS
        AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
        VitsModel.from_pretrained(model_name, cache_dir=model_path)
        print(f"LOG: MMS-TTS {lang_code} téléchargé dans {model_path}")
    else:
        print(f"LOG: MMS-TTS {lang_code} déjà présent dans {model_path}")

    return model_path

def speak_mms(text, lang_code):
    """
    Synthèse vocale MMS-TTS utilisant VitsModel et AutoTokenizer.
    """
    model_id = MMS_TTS_MODELS[lang_code]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        st.info(f"Génération audio pour {lang_code} avec {model_id}...")
        # Chargement du modèle Vits
        model = VitsModel.from_pretrained(model_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Tokenisation du texte
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # Génération de la voix
        with torch.no_grad():
            waveform = model(**inputs).waveform

        # Création du fichier temporaire et lecture audio
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sampling_rate = model.config.sampling_rate
        sf.write(tmp_wav, waveform.squeeze().cpu().numpy(), sampling_rate)
        st.audio(tmp_wav)
        print(f"LOG: Lecture audio terminée, fichier temporaire: {tmp_wav}")

        os.unlink(tmp_wav)

    except Exception as e:
        st.error(f"Erreur lors de la synthèse vocale : {e}")
        print(f"ERROR: {e}")




def speak_japanese_melo(text):
    """
    Synthèse vocale Japonais via script externe MeloTTS
    """
    script_path = os.path.join(base_path, "generate_tts_jp.py")
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    try:
        cmd = ["python", script_path, json.dumps({'text': text, 'output_path': tmp_wav})]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0 and "Success" in result.stdout:
            st.audio(tmp_wav)
            st.success("✅ Synthèse vocale japonaise réussie")
        else:
            st.error(f"Erreur script Japonais : {result.stderr}")
        os.unlink(tmp_wav)
    except Exception as e:
        st.error(f"Erreur TTS Japonais : {e}")




def speak_chinese_melo(text):
    script_path = os.path.join(base_path, "generate_tts_ch.py")
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    try:
        # Forcer CPU et désactiver MPS sur Mac
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""        # Désactive CUDA
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0" # Désactive MPS fallback

        cmd = ["python", script_path, json.dumps({'text': text, 'output_path': tmp_wav})]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        if result.returncode == 0 and "Success" in result.stdout:
            st.audio(tmp_wav)
            st.success("✅ Synthèse vocale chinoise réussie")
        else:
            st.error(f"Erreur TTS chinois : {result.stderr}")
        os.unlink(tmp_wav)
    except Exception as e:
        st.error(f"Erreur TTS Chinois : {e}")





# =======================
# FONCTIONS DE LOGIQUE
# =======================

def detect_silence(audio_chunk, threshold=SILENCE_THRESHOLD):
    rms = np.sqrt(np.mean(audio_chunk ** 2))
    return rms < threshold

def record_with_silence_detection(duration=MAX_RECORDING_TIME, sample_rate=SAMPLE_RATE):
    print("LOG: Démarrage de l'enregistrement audio...")
    audio_queue = queue.Queue()
    recording = []
    def callback(indata, frames, time, status):
        if status:
            print(f"LOG: Audio status: {status}")
        audio_queue.put(indata.copy())
    with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate):
        start = time.time()
        silence_start = None
        while True:
            try:
                chunk = audio_queue.get(timeout=0.1)
                recording.append(chunk)
                elapsed = time.time() - start
                if elapsed >= duration:
                    print("LOG: Durée maximale atteinte")
                    break
                if detect_silence(chunk):
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= SILENCE_DURATION:
                        print("LOG: Silence détecté, arrêt de l'enregistrement")
                        break
                else:
                    silence_start = None
            except queue.Empty:
                continue
    if recording:
        audio_data = np.concatenate(recording, axis=0)
        print(f"LOG: Enregistrement terminé, {len(audio_data)} échantillons capturés")
        return (audio_data * 32767).astype(np.int16)
    print("LOG: Aucun audio enregistré")
    return None

def transcribe(lang_name):
    print(f"LOG: Début de la transcription pour {lang_name}")
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    # Enregistrement audio avec détection du silence
    audio = record_with_silence_detection()
    if audio is None:
        st.warning("Aucun audio détecté")
        return ""

    # Sauvegarde en WAV temporaire
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    # Cas particulier : Japonais ou Mandarin -> toujours Whisper
    if lang_name in ["Japonais", "Mandarin"]:
        print(f"LOG: Transcription {lang_name} avec Whisper (pas de modèle Vosk ou par préférence)")
        model = load_whisper_model()
        lang_code = "ja" if lang_name == "Japonais" else "zh"
        result = model.transcribe(wav_path, language=lang_code)
        os.unlink(wav_path)
        transcription = result["text"]
        print(f"LOG: Transcription Whisper ({lang_code}) terminée: {transcription}")
        return transcription

    # Pour les autres langues -> essai Vosk
    vosk_model = download_vosk_model(lang_name)
    if vosk_model:
        print(f"LOG: Transcription {lang_name} avec Vosk")
        rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
        rec.SetWords(True)
        results = []
        with wave.open(wav_path, "rb") as wf:
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    if 'text' in res:
                        results.append(res['text'])
            final = json.loads(rec.FinalResult())
            if 'text' in final:
                results.append(final['text'])
        os.unlink(wav_path)
        transcription = " ".join(results)
        print(f"LOG: Transcription Vosk terminée: {transcription}")
        return transcription

    # Fallback Whisper si Vosk indisponible
    print(f"LOG: Transcription {lang_name} avec Whisper (fallback)")
    model = load_whisper_model()
    code = LANGUAGES_DATA[lang_name]["seamless_code"].strip("_")
    result = model.transcribe(wav_path, language=code)
    os.unlink(wav_path)
    transcription = result["text"]
    print(f"LOG: Transcription Whisper (fallback) terminée: {transcription}")
    return transcription

def translate(text, src, tgt, model, processor):
    print(f"LOG: Traduction de {src} vers {tgt}")
    inputs = processor(text=text.strip(), src_lang=LANGUAGES_DATA[src]["seamless_code"], return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs,
                                tgt_lang=LANGUAGES_DATA[tgt]["seamless_code"],
                                generate_speech=False,
                                max_length=256)
    translated = processor.decode(output.sequences[0].tolist(), skip_special_tokens=True)
    clean = re.sub(r'__[\w_]+__', '', translated).strip()
    print(f"LOG: Traduction terminée: {clean}")
    return clean

# =======================
# INTERFACE STREAMLIT AMÉLIORÉE
# =======================

st.title("🧠 Interprète Multilingue (100% Offline)")
st.markdown("**Version améliorée avec détection de silence et TTS optimisé**")
st.markdown("---")

# Configuration des langues
col1, col2 = st.columns(2)
with col1:
    input_lang = st.selectbox("🎤 Langue de l'interlocuteur :", list(LANGUAGES_DATA.keys()), index=0)
with col2:
    target_lang = st.selectbox("🔤 Langue de sortie :", list(LANGUAGES_DATA.keys()), index=1)

# Vérification que les langues sont différentes
if input_lang == target_lang:
    st.warning("⚠️ Les langues source et cible sont identiques")

# Bouton pour valider et charger les modèles
if st.button("✅ Valider et charger modèles"):
    st.session_state.selected_pair = (input_lang, target_lang)
    st.info(f"Téléchargement des modèles pour {input_lang} et {target_lang}...")
    download_vosk_model(input_lang)
    download_mms_model(LANGUAGES_DATA[input_lang]["tts_lang_code"])
    download_mms_model(LANGUAGES_DATA[target_lang]["tts_lang_code"])
    st.success("✅ Modèles téléchargés et prêts !")

st.markdown("---")

# === ÉTAPE 1: TRANSCRIPTION ===
if "selected_pair" in st.session_state:
    st.subheader("1. 🎤 Transcription avec détection de silence")
    with st.expander("ℹ️ Paramètres d'écoute"):
        st.write(f"• Durée maximale : {MAX_RECORDING_TIME} s")
        st.write(f"• Arrêt automatique : {SILENCE_DURATION} s de silence")
        st.write(f"• Seuil de silence : {SILENCE_THRESHOLD}")

    if st.button("🎤 Démarrer écoute intelligente"):
        with st.spinner("Enregistrement en cours..."):
            transcription = transcribe(st.session_state.selected_pair[0])
            if transcription:
                st.session_state.last_transcription = transcription
                st.success(f"✅ Transcription réussie ({st.session_state.selected_pair[0]})")
                st.write(f"**Texte transcrit :** {transcription}")
            else:
                st.error("❌ Aucun texte détecté")

# === ÉTAPE 2: TRADUCTION ===
if st.session_state.get("last_transcription"):
    st.markdown("---")
    st.subheader("2. 🌍 Traduction")
    st.write(f"Traduction : **{st.session_state.selected_pair[0]} → {st.session_state.selected_pair[1]}**")
    st.write(f"**Texte à traduire :** {st.session_state.last_transcription}")

    if st.button("🌍 Traduire"):
        with st.spinner("Traduction en cours..."):
            model, processor = get_translator_pipeline()
            translated_text = translate(
                st.session_state.last_transcription,
                st.session_state.selected_pair[0],
                st.session_state.selected_pair[1],
                model, processor
            )
            if translated_text:
                st.session_state.translated = translated_text
                st.success("✅ Traduction réussie")
                st.write(f"**Traduction :** {translated_text}")
            else:
                st.error("❌ Échec de la traduction")

# === ÉTAPE 3: SYNTHÈSE VOCALE ===
if st.session_state.get("translated"):
    st.markdown("---")
    st.subheader("3. 🔊 Lecture audio de la traduction")
    st.write(f"**Texte à prononcer :** {st.session_state.translated}")
    target_tts_code = LANGUAGES_DATA[st.session_state.selected_pair[1]]["tts_lang_code"]
    if st.button("🔊 Lire traduction"):
        if target_tts_code == "JA":
            speak_japanese_melo(st.session_state.translated)
        elif target_tts_code == "ZH":
            speak_chinese_melo(st.session_state.translated)
        else:
            speak_mms(st.session_state.translated, target_tts_code)




# === ÉTAPE 4: VOTRE RÉPONSE ===
st.markdown("---")
st.subheader("4. 📝 Votre réponse")
st.info(f"Écrivez votre réponse en **{st.session_state.get('selected_pair', ('', ''))[1]}**")

user_response = st.text_area("Tapez votre réponse ici...", key="user_response_text")
if st.button("🔁 Traduire et prononcer votre réponse"):
    if user_response.strip():
        model, processor = get_translator_pipeline()
        translated_response = translate(
            user_response,
            st.session_state.selected_pair[1],
            st.session_state.selected_pair[0],
            model, processor
        )
        st.session_state.translated_response = translated_response
        st.write(f"**Réponse traduite :** {translated_response}")

        # Détection de la langue pour TTS
        target_lang_code = LANGUAGES_DATA[st.session_state.selected_pair[0]]["tts_lang_code"]

        if target_lang_code == "JA":
            # Utilisation de MeloTTS japonais via script externe
            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            data = {"text": translated_response, "output_path": tmp_wav}
            try:
                result = subprocess.run(
                    ["python", "generate_tts_jp.py", json.dumps(data)],
                    capture_output=True, text=True
                )
                if result.returncode == 0 and "Success" in result.stdout:
                    st.audio(tmp_wav)
                    st.success("✅ Synthèse vocale japonaise réussie")
                else:
                    st.error(f"Erreur TTS japonais : {result.stderr}")
                os.unlink(tmp_wav)
            except Exception as e:
                st.error(f"Erreur lors de l'appel à MeloTTS japonais : {e}")

        elif target_lang_code == "ZH":
            # Utilisation de MeloTTS chinois via script externe (CPU forcé, MPS désactivé)
            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            data = {"text": translated_response, "output_path": tmp_wav}
            try:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ""        # Désactive CUDA
                env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0" # Désactive MPS fallback
                result = subprocess.run(
                    ["python", "generate_tts_ch.py", json.dumps(data)],
                    capture_output=True, text=True,
                    env=env
                )
                if result.returncode == 0 and "Success" in result.stdout:
                    st.audio(tmp_wav)
                    st.success("✅ Synthèse vocale chinoise réussie")
                else:
                    st.error(f"Erreur TTS chinois : {result.stderr}")
                os.unlink(tmp_wav)
            except Exception as e:
                st.error(f"Erreur lors de l'appel à MeloTTS chinois : {e}")

        else:
            # Synthèse vocale MMS pour les autres langues
            speak_mms(translated_response, target_lang_code)
    else:
        st.warning("Veuillez taper votre réponse avant de continuer.")






# === DEBUG / INFO SESSION ===
with st.expander("🔧 Informations de débogage"):
    st.write(st.session_state)

# === RÉINITIALISATION ===
st.markdown("---")
if st.button("🔄 Réinitialiser la session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Session réinitialisée")
    st.experimental_rerun()