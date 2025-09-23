import streamlit as st
from transformers import AutoProcessor, SeamlessM4TModel
import whisper
import re
import tempfile, os, torch, wave, sounddevice as sd
import json
import numpy as np
from vosk import Model, KaldiRecognizer
from TTS.api import TTS
import threading
import queue
import time

# =======================
# CONFIG OFFLINE
# =======================
LANGUAGES_TO_CODES = {
    "Français": "__fra__",
    "Anglais": "__eng__",
    "Japonais": "__jpn__"
}
MODEL_LOCAL_DIR = "/Users/macbook/NeuraFox/Traduction-App/hf-seamless-m4t-medium"

VOSK_MODELS = {
    "Français": "/Users/macbook/NeuraFox/Traduction-App/vosk-model-fr",
    "Anglais": "/Users/macbook/NeuraFox/Traduction-App/vosk-model-en",
}

TTS_MODELS = {
    "Français": "tts_models/fr/css10/vits",            
    "Anglais": "tts_models/en/ljspeech/vits",         
    "Japonais": "/Users/macbook/NeuraFox/Traduction-App/models/tts/melotts-japanese"  
}


# Paramètres pour la détection de silence
SILENCE_THRESHOLD = 0.01  # Seuil de détection du silence (amplitude)
SILENCE_DURATION = 2.0    # Durée de silence pour arrêter l'enregistrement (secondes)
MAX_RECORDING_TIME = 60.0 # Durée maximale d'enregistrement (secondes)
SAMPLE_RATE = 16000       # Fréquence d'échantillonnage

# =======================
# CHARGEMENT DES MODÈLES
# =======================

@st.cache_resource
def get_translator_pipeline():
    try:
        processor = AutoProcessor.from_pretrained(MODEL_LOCAL_DIR, local_files_only=True)
        model = SeamlessM4TModel.from_pretrained(MODEL_LOCAL_DIR, local_files_only=True)
        return model, processor
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle de traduction : {e}")
        st.stop()

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small", device="cpu")

@st.cache_resource
def load_vosk_model(lang_name):
    path = VOSK_MODELS.get(lang_name)
    if not path or not os.path.exists(path):
        st.error(f"Modèle Vosk introuvable pour {lang_name} : {path}")
        return None
    return Model(path)

@st.cache_resource
def load_tts_model(lang_name):
    try:
        model_path = TTS_MODELS[lang_name]
        if lang_name == "Japonais":
            from melo.api import TTS as MeloTTS
            model = MeloTTS(language='JP', device='cpu')
            # Récupération de l'ID du speaker JP
            model.speaker_id = model.hps.data.spk2id['JP']
            return model
        else:
            from TTS.api import TTS
            return TTS(model_path, gpu=False)
    except Exception as e:
        st.error(f"Erreur lors du chargement TTS pour {lang_name} : {e}")
        return None


# =======================
# FONCTIONS AMÉLIORÉES
# =======================

def detect_silence(audio_chunk, threshold=SILENCE_THRESHOLD):
    """Détecte si un chunk audio contient du silence"""
    rms = np.sqrt(np.mean(audio_chunk ** 2))
    return rms < threshold

def record_with_silence_detection(duration=MAX_RECORDING_TIME, sample_rate=SAMPLE_RATE):
    """
    Enregistre l'audio avec détection automatique de silence
    S'arrête après 2 secondes de silence ou après la durée maximale
    """
    audio_queue = queue.Queue()
    recording = []
    
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        audio_queue.put(indata.copy())
    
    # Démarrage de l'enregistrement
    with sd.InputStream(callback=audio_callback, 
                       channels=1, 
                       samplerate=sample_rate, 
                       dtype='float32'):
        
        start_time = time.time()
        silence_start = None
        
        st.info("🎤 Enregistrement en cours... Parlez maintenant ! (Max 60s, arrêt automatique après 2s de silence)")
        
        # Placeholder pour afficher le temps restant
        time_placeholder = st.empty()
        
        while True:
            try:
                # Récupération des données audio (timeout pour éviter le blocage)
                chunk = audio_queue.get(timeout=0.1)
                recording.append(chunk)
                
                # Vérification du temps écoulé
                elapsed_time = time.time() - start_time
                remaining_time = max(0, duration - elapsed_time)
                time_placeholder.info(f"⏱️ Temps restant: {remaining_time:.1f}s")
                
                # Arrêt si durée maximale atteinte
                if elapsed_time >= duration:
                    st.info("⏰ Durée maximale atteinte (60s)")
                    break
                
                # Détection de silence
                if detect_silence(chunk):
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= SILENCE_DURATION:
                        st.info("🤫 Silence détecté - Arrêt de l'enregistrement")
                        break
                else:
                    silence_start = None  # Reset du compteur de silence
                    
            except queue.Empty:
                # Pas de nouvelles données, continuer
                continue
            except Exception as e:
                st.error(f"Erreur lors de l'enregistrement : {e}")
                break
    
    time_placeholder.empty()
    
    if recording:
        # Concaténation de tous les chunks
        audio_data = np.concatenate(recording, axis=0)
        # Conversion en int16 pour la compatibilité
        audio_int16 = (audio_data * 32767).astype(np.int16)
        return audio_int16
    else:
        return None

def transcribe_offline_improved(lang_name):
    """Version améliorée de la transcription avec détection de silence"""
    try:
        # Enregistrement avec détection de silence
        recording = record_with_silence_detection()
        
        if recording is None or len(recording) == 0:
            st.warning("Aucun audio enregistré")
            return ""
        
        # Sauvegarde temporaire
        wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(recording.tobytes())

        text = ""
        
        # Japonais → Whisper
        if lang_name == "Japonais":
            model = load_whisper_model()
            result = model.transcribe(wav_path, language="ja")
            text = result["text"].strip()
        else:
            # FR ou EN → Vosk
            vosk_model = load_vosk_model(lang_name)
            if vosk_model is None:
                return ""
                
            with wave.open(wav_path, "rb") as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
                    st.error("Format audio incompatible avec Vosk")
                    return ""
                
                rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
                rec.SetWords(True)
                
                results = []
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        if 'text' in result:
                            results.append(result['text'])
                
                # Résultat final
                final_result = json.loads(rec.FinalResult())
                if 'text' in final_result:
                    results.append(final_result['text'])
                
                text = ' '.join(results).strip()

        # Nettoyage du fichier temporaire
        try:
            os.unlink(wav_path)
        except:
            pass
            
        return text

    except Exception as e:
        st.error(f"Erreur lors de la transcription : {e}")
        return ""

def translate_text(text, source_lang, target_lang, model, processor):
    """Traduit le texte de source_lang vers target_lang en utilisant SeamlessM4T"""
    try:
        if not text or not text.strip():
            return ""
            
        src_code = LANGUAGES_TO_CODES[source_lang]
        tgt_code = LANGUAGES_TO_CODES[target_lang]
        
        # Préparation des inputs avec le processeur pour le texte et la langue source
        text_inputs = processor(text=text.strip(), src_lang=src_code, return_tensors="pt")
        
        # Génération
        with torch.no_grad():
            output_object = model.generate(
                **text_inputs,
                tgt_lang=tgt_code,
                generate_speech=False,
                max_length=256,
                num_beams=5,
                early_stopping=True,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Accéder aux séquences générées
        generated_tokens = output_object.sequences 
        
        # Décodage
        translated_text = processor.decode(generated_tokens[0].tolist(), skip_special_tokens=True)
        
        # Nettoyage des balises résiduelles
        clean_text = re.sub(r'__[\w_]+__', '', translated_text).strip()

        # Vérification que la traduction n'est pas vide
        if not clean_text:
            st.warning("La traduction est vide après nettoyage")
            return translated_text
        
        return clean_text

    except Exception as e:
        st.error(f"Erreur lors de la traduction : {e}")
        return ""


def speak_text_improved(text, lang_name):
    if not text or not text.strip():
        st.warning("Texte vide, synthèse vocale ignorée")
        return False
        
    try:
        tts = load_tts_model(lang_name)
        if tts is None:
            st.error(f"❌ Modèle TTS non disponible pour {lang_name}")
            return False
        
        clean_text = text.strip()
        if len(clean_text) > 500:
            clean_text = clean_text[:500] + "..."
        
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        
        with st.spinner(f"Génération audio en {lang_name}..."):
            if lang_name == "Japonais":
                # MeloTTS
                tts.tts_to_file(clean_text, tts.speaker_id, out_path, speed=1.0)
            else:
                # TTS standard
                tts.tts_to_file(text=clean_text, file_path=out_path)
        
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            st.audio(out_path, format="audio/wav")
            st.success(f"✅ Synthèse vocale réussie en {lang_name}")
            try: os.unlink(out_path)
            except: pass
            return True
        else:
            st.error("❌ Fichier audio généré vide ou inexistant")
            return False

    except Exception as e:
        st.error(f"❌ Erreur TTS pour {lang_name} : {e}")
        return False


# =======================
# INTERFACE STREAMLIT
# =======================

st.title("🧠 Interprète Multilingue (100% Offline)")
st.markdown("**Version améliorée avec détection de silence et TTS optimisé**")
st.markdown("---")

# Configuration des langues
col1, col2 = st.columns(2)
with col1:
    input_lang = st.selectbox("🎤 Langue de l'interlocuteur :", ["Français", "Anglais", "Japonais"], index=0)
with col2:
    target_lang = st.selectbox("🔤 Langue de sortie :", list(LANGUAGES_TO_CODES.keys()), index=1)

# Vérification que les langues sont différentes
if input_lang == target_lang:
    st.warning("⚠️ Les langues source et cible sont identiques")

# Charger le modèle de traduction
try:
    model, processor = get_translator_pipeline()
    st.success("✅ Modèle de traduction chargé")
except Exception as e:
    st.error(f"❌ Impossible de charger le modèle de traduction: {e}")
    st.stop()

st.markdown("---")

# === ÉTAPE 1: TRANSCRIPTION AMÉLIORÉE ===
st.subheader("1. 🎤 Écoute et Transcription (avec détection de silence)")

# Informations sur les paramètres d'écoute
with st.expander("ℹ️ Paramètres d'écoute"):
    st.write(f"• **Durée maximale :** {MAX_RECORDING_TIME} secondes")
    st.write(f"• **Arrêt automatique :** {SILENCE_DURATION} secondes de silence")
    st.write(f"• **Seuil de silence :** {SILENCE_THRESHOLD}")
    st.write("• L'enregistrement s'arrête dès qu'un silence de 2 secondes est détecté")

if st.button("🎤 Démarrer écoute intelligente", type="primary"):
    with st.spinner("Préparation de l'écoute..."):
        transcribed_text = transcribe_offline_improved(input_lang)
        
        if transcribed_text:
            st.session_state.last_transcription = transcribed_text
            st.session_state.source_lang = input_lang
            st.success(f"✅ Transcription réussie ({input_lang})")
            st.write(f"**Texte transcrit :** {transcribed_text}")
            st.info(f"📝 Longueur : {len(transcribed_text)} caractères")
        else:
            st.error("❌ Échec de la transcription - Aucun texte détecté")
            if 'last_transcription' in st.session_state:
                del st.session_state.last_transcription

# === ÉTAPE 2: TRADUCTION ===
if st.session_state.get("last_transcription"):
    st.markdown("---")
    st.subheader("2. 🌍 Traduction")
    
    st.info(f"Traduction : **{st.session_state.source_lang}** → **{target_lang}**")
    st.write(f"**Texte à traduire :** {st.session_state.last_transcription}")
    
    if st.button("🌍 Traduire", type="primary"):
        with st.spinner("Traduction en cours..."):
            translated_text = translate_text(
                st.session_state.last_transcription,
                st.session_state.source_lang,
                target_lang,
                model,
                processor
            )
            
            if translated_text:
                st.session_state.translated = translated_text
                st.session_state.target_lang = target_lang
                st.success(f"✅ Traduction réussie")
                st.write(f"**Traduction ({target_lang}) :** {translated_text}")
                st.info(f"📝 Longueur : {len(translated_text)} caractères")
            else:
                st.error("❌ Échec de la traduction")

# === ÉTAPE 3: SYNTHÈSE VOCALE AMÉLIORÉE (traduction de l'interlocuteur) ===
if st.session_state.get("translated"):
    st.markdown("---")
    st.subheader("3. 🔊 Synthèse Vocale Améliorée")
    
    current_translation = st.session_state.translated
    current_target_lang = st.session_state.get('target_lang', target_lang)
    
    st.write(f"**Texte à prononcer :** {current_translation}")
    st.write(f"**Langue :** {current_target_lang}")
    
    # Informations sur le modèle TTS
    tts_model_info = TTS_MODELS.get(current_target_lang, "Modèle non disponible")
    st.caption(f"Modèle TTS : {tts_model_info}")
    
    if current_target_lang not in TTS_MODELS:
        st.error(f"❌ Aucun modèle TTS disponible pour {current_target_lang}")
    else:
        if st.button("🔊 Prononcer la traduction", type="primary"):
            success = speak_text_improved(current_translation, current_target_lang)
            if success:
                st.balloons()  # Effet visuel de succès

# === ÉTAPE 4: VOTRE RÉPONSE (nouvelle section) ===
st.markdown("---")
st.subheader("4. 📝 Votre Réponse")
st.info(f"Écrivez votre réponse en **{target_lang}** pour la traduire et la prononcer en **{input_lang}**.")

user_response_text = st.text_area("Tapez votre réponse ici...", key="user_response_text")

if st.button("Traduire et parler (votre réponse)", type="secondary"):
    if user_response_text:
        with st.spinner("Traduction et génération audio de votre réponse..."):
            # Traduire de la langue cible à la langue de l'interlocuteur
            translated_response = translate_text(
                user_response_text,
                target_lang,
                input_lang,
                model,
                processor
            )
            
            if translated_response:
                st.success("✅ Traduction de votre réponse réussie.")
                st.write(f"**Votre réponse traduite ({input_lang}) :** {translated_response}")
                
                # Prononcer la réponse traduite
                speak_text_improved(translated_response, input_lang)
            else:
                st.error("❌ Échec de la traduction de votre réponse.")
    else:
        st.warning("Veuillez taper votre réponse avant de continuer.")


# === DEBUG INFO ===
with st.expander("🔧 Informations de débogage"):
    st.write("**État de la session :**")
    for key, value in st.session_state.items():
        st.write(f"- {key}: {value}")
    
    st.write("**Paramètres audio :**")
    st.write(f"- Fréquence d'échantillonnage: {SAMPLE_RATE} Hz")
    st.write(f"- Seuil de silence: {SILENCE_THRESHOLD}")
    st.write(f"- Durée de silence: {SILENCE_DURATION}s")
    st.write(f"- Durée max: {MAX_RECORDING_TIME}s")
    
    st.write("**Chemins des modèles :**")
    st.write(f"- Traduction: {MODEL_LOCAL_DIR}")
    st.write(f"- Vosk FR: {VOSK_MODELS.get('Français', 'N/A')}")
    st.write(f"- Vosk EN: {VOSK_MODELS.get('Anglais', 'N/A')}")
    
    st.write("**Modèles TTS :**")
    for lang, model in TTS_MODELS.items():
        st.write(f"- {lang}: {model}")

# === RÉINITIALISATION ===
st.markdown("---")
if st.button("🔄 Réinitialiser la session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Session réinitialisée")
    st.rerun()
