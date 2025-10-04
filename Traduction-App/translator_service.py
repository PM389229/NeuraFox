# translator_service.py
import re
import tempfile, os, torch, wave, sounddevice as sd
import json
import numpy as np
import subprocess # Nécessaire pour appeler melo_tts_service.py
import queue
import time
import soundfile as sf
import streamlit as st
import sys
import logging

from vosk import KaldiRecognizer # Import explicite pour éviter l'erreur dans un environnement strict
from transformers import VitsModel, AutoTokenizer # Nécessaire pour _speak_mms
from model_manager import ModelManager
from config import LANGUAGES_DATA, SAMPLE_RATE, SILENCE_THRESHOLD, SILENCE_DURATION, MAX_RECORDING_TIME, MMS_TTS_MODELS

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='LOG: %(message)s')

class TranslatorService:
    def __init__(self):
        self.model_manager = ModelManager()

    # --- ASR / Transcription ---
    def detect_silence(self, audio_chunk):
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        return rms < SILENCE_THRESHOLD

    def record_with_silence_detection(self):
        logging.info("Démarrage de l'enregistrement audio...")
        audio_queue = queue.Queue()
        recording = []
        
        # Définit le callback audio
        def callback(indata, frames, time, status):
            if status:
                logging.warning(f"Audio status: {status}")
            audio_queue.put(indata.copy())
            
        # Démarre le flux audio
        with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE):
            start = time.time()
            silence_start = None
            
            # Boucle d'enregistrement et de détection de silence
            while True:
                try:
                    # Traite les chunks audio toutes les 100ms
                    chunk = audio_queue.get(timeout=0.1) 
                    recording.append(chunk)
                    elapsed = time.time() - start
                    
                    if elapsed >= MAX_RECORDING_TIME:
                        logging.info("Durée maximale atteinte")
                        break
                        
                    if self.detect_silence(chunk):
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start >= SILENCE_DURATION:
                            logging.info("Silence détecté, arrêt de l'enregistrement")
                            break
                    else:
                        silence_start = None
                except queue.Empty:
                    continue
                    
        if recording:
            audio_data = np.concatenate(recording, axis=0)
            logging.info(f"Enregistrement terminé, {len(audio_data)} échantillons capturés")
            # Convertit en Int16 et s'assure que les valeurs sont dans la plage correcte
            return (audio_data * 32767).astype(np.int16)
        logging.info("Aucun audio enregistré")
        return None

    def transcribe(self, lang_name):
        wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        audio = self.record_with_silence_detection()
        if audio is None:
            if os.path.exists(wav_path):
                 os.unlink(wav_path) # Nettoyage si aucun enregistrement
            return ""

        # Sauvegarde en WAV temporaire
        try:
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio.tobytes())
        except Exception as e:
             st.error(f"Erreur lors de l la sauvegarde du WAV: {e}")
             if os.path.exists(wav_path): os.unlink(wav_path)
             return ""


        transcription = ""
        
        # 1. Cas spécial : Japonais ou Mandarin -> Whisper
        if lang_name in ["Japonais", "Mandarin"]:
            whisper_model = self.model_manager.get_whisper_model()
            if whisper_model:
                lang_code = "ja" if lang_name == "Japonais" else "zh"
                try:
                    # Utiliser le chemin WAV pour la transcription Whisper
                    result = whisper_model.transcribe(wav_path, language=lang_code)
                    transcription = result["text"]
                except Exception as e:
                    st.error(f"Erreur Whisper: {e}")
            else:
                 st.error("Whisper n'a pas pu être chargé pour cette langue.")
        
        # 2. Vosk pour les autres langues
        else:
            vosk_model = self.model_manager.download_vosk_model(lang_name)
            if vosk_model:
                # La logique d'exécution de Vosk doit être adaptée pour utiliser le fichier WAV
                # Pour garder la cohérence avec l'architecture Vosk (KaldiRecognizer),
                # nous allons lire les frames du fichier WAV et les passer à KaldiRecognizer.
                rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
                rec.SetWords(True)
                results = []
                try:
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
                    transcription = " ".join(results)
                except Exception as e:
                     st.warning(f"Erreur Vosk lors de la reconnaissance : {e}")
            
            # 3. Fallback Whisper si Vosk échoue ou est vide
            if not transcription.strip():
                st.warning(f"Vosk n'a pas pu transcrire ou n'est pas disponible pour {lang_name}. Tentative de fallback avec Whisper...")
                whisper_model = self.model_manager.get_whisper_model(model_size="tiny") # Utilise tiny pour la vitesse
                if whisper_model:
                    code = LANGUAGES_DATA[lang_name]["seamless_code"].strip("_")
                    try:
                        result = whisper_model.transcribe(wav_path, language=code)
                        transcription = result["text"]
                    except Exception as e:
                        st.error(f"Erreur Whisper Fallback: {e}")

        os.unlink(wav_path)
        return transcription.strip()

    # --- Traduction ---
    def translate(self, text, src, tgt):
        model, processor = self.model_manager.get_translator_pipeline()
        
        if not model or not processor:
             st.error("Modèle de traduction non disponible.")
             return ""

        # Utilisez les codes Seamless pour la traduction
        src_seamless_code = LANGUAGES_DATA[src]["seamless_code"]
        tgt_seamless_code = LANGUAGES_DATA[tgt]["seamless_code"]

        inputs = processor(text=text.strip(), src_lang=src_seamless_code, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs,
                                     tgt_lang=tgt_seamless_code,
                                     generate_speech=False,
                                     max_length=256)
        translated = processor.decode(output.sequences[0].tolist(), skip_special_tokens=True)
        # Nettoyage des tokens spéciaux Seamless (ex: __eng__)
        clean = re.sub(r'__[\w_]+__', '', translated).strip()
        return clean

    # --- Synthèse Vocale (TTS) ---
    def speak(self, text, lang_name):
        target_tts_code = LANGUAGES_DATA[lang_name]["tts_lang_code"]

        if target_tts_code in ["JA", "ZH"]:
            self._speak_melo(text, target_tts_code)
        else:
            self._speak_mms(text, target_tts_code)

    def _speak_mms(self, text, lang_code):
        """ Synthèse vocale MMS-TTS utilisant VitsModel et AutoTokenizer. """
        model_id = MMS_TTS_MODELS[lang_code] 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            st.info(f"Génération audio pour {lang_code} avec {model_id}...")
            
            # Utilisation de la méthode from_pretrained pour charger le modèle directement
            model = VitsModel.from_pretrained(model_id).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            inputs = tokenizer(text, return_tensors="pt").to(device)

            with torch.no_grad():
                waveform = model(**inputs).waveform

            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            sampling_rate = model.config.sampling_rate
            sf.write(tmp_wav, waveform.squeeze().cpu().numpy(), sampling_rate)
            st.audio(tmp_wav, sample_rate=sampling_rate)
            os.unlink(tmp_wav)

        except Exception as e:
            st.error(f"❌ Erreur lors de la synthèse vocale MMS : {e}")
            logging.error(f"Erreur MMS TTS: {e}")

    def _speak_melo(self, text, lang_code):
        """
        Synthèse vocale pour JA/ZH en appelant le script externe melo_tts_service.py via subprocess.
        """
        try:
            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            
            # Préparation des données JSON à passer au script externe
            input_data = {
                'text': text,
                'output_path': tmp_wav,
                'lang': lang_code
            }
            json_input = json.dumps(input_data)
            
            # Chemin absolu vers le script externe (basé sur le répertoire du script actuel)
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "melo_tts_service.py")
            
            # Appel du script externe
            command = [
                sys.executable,
                script_path,
                json_input
            ]
            
            st.info(f"Génération audio pour {lang_code} avec MeloTTS (subprocess)...")
            
            # Exécution de la commande
            # check=True lève une exception si le subprocess échoue (return code != 0)
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True 
            )
            
            if "Success" in result.stdout:
                # MeloTTS utilise typiquement 22050 Hz.
                st.audio(tmp_wav, sample_rate=22050) 
            else:
                st.error(f"❌ Échec de la synthèse vocale MeloTTS. Erreur: {result.stderr}")
                
            os.unlink(tmp_wav)
            
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Erreur d'exécution de MeloTTS. Code de retour: {e.returncode}. Sortie d'erreur: {e.stderr}")
            logging.error(f"MeloTTS Subprocess Error (Code: {e.returncode}): {e.stderr}")
        except Exception as e:
            st.error(f"❌ Erreur générale lors de l'appel de MeloTTS: {e}")
            logging.error(f"MeloTTS Error: {e}")