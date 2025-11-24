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
# Changement: Passage au niveau DEBUG pour un diagnostic plus précis
logging.basicConfig(level=logging.DEBUG, format='LOG: %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TranslatorService:
    def __init__(self):
        self.model_manager = ModelManager()

    # --- ASR / Transcription ---
    def detect_silence(self, audio_chunk):
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        # Log ajouté: Détail du niveau audio pour le VAD
        logger.debug(f"RMS: {rms:.4f}, Seuil: {SILENCE_THRESHOLD}")
        return rms < SILENCE_THRESHOLD

    def record_with_silence_detection(self):
        logger.info("Démarrage de l'enregistrement audio...")
        audio_queue = queue.Queue()
        recording = []
        
        # Définit le callback audio
        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio status: {status}")
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
                        logger.info("Durée maximale atteinte")
                        break
                        
                    if self.detect_silence(chunk):
                        if silence_start is None:
                            silence_start = time.time()
                            # Log ajouté
                            logger.debug("Début de la détection de silence.")
                        elif time.time() - silence_start >= SILENCE_DURATION:
                            logger.info("Silence détecté, arrêt de l'enregistrement")
                            break
                    else:
                        # Log ajouté
                        if silence_start is not None:
                            logger.debug("Parole détectée, réinitialisation du silence.")
                        silence_start = None
                except queue.Empty:
                    continue
                    
        if recording:
            audio_data = np.concatenate(recording, axis=0)
            logger.info(f"Enregistrement terminé, {len(audio_data)} échantillons capturés")
            # Convertit en Int16 et s'assure que les valeurs sont dans la plage correcte
            return (audio_data * 32767).astype(np.int16)
        logger.info("Aucun audio enregistré")
        return None

    def transcribe(self, lang_name):
        # Log ajouté
        logger.info(f"Début de la transcription pour la langue: {lang_name}")
        wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        audio = self.record_with_silence_detection()
        if audio is None:
            # Log ajouté
            logger.info("Aucun audio enregistré. Fin de la transcription.")
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
            # Log ajouté
            logger.info(f"Fichier WAV temporaire créé à: {wav_path}")
        except Exception as e:
            st.error(f"Erreur lors de l la sauvegarde du WAV: {e}")
            # Log ajouté
            logger.error(f"Erreur de sauvegarde WAV: {e}")
            if os.path.exists(wav_path): os.unlink(wav_path)
            return ""


        transcription = ""
        
        # 1. Cas spécial : Japonais ou Mandarin -> Whisper
        if lang_name in ["Japonais", "Mandarin"]:
            # Log ajouté
            logger.info(f"Utilisation de Whisper pour {lang_name}.")
            whisper_model = self.model_manager.get_whisper_model()
            if whisper_model:
                lang_code = "ja" if lang_name == "Japonais" else "zh"
                # Log ajouté
                logger.info(f"Code langue Whisper: {lang_code}. Fichier à transcrire: {wav_path}")
                try:
                    # Utiliser le chemin WAV pour la transcription Whisper
                    result = whisper_model.transcribe(wav_path, language=lang_code)
                    transcription = result["text"]
                    # Log ajouté
                    logger.info(f"Transcription Whisper réussie: '{transcription}'")
                except Exception as e:
                    # Amélioration du message d'erreur pour aider au diagnostic FFmpeg
                    if "WinError 2" in str(e):
                        error_msg = f"❌ Erreur Whisper: Le programme externe (probablement FFmpeg) est introuvable. Veuillez vérifier votre PATH. Détail: {e}"
                        st.error(error_msg)
                        logger.error(error_msg)
                    else:
                        st.error(f"Erreur Whisper: {e}")
                        # Log ajouté
                        logger.error(f"Erreur lors de la transcription Whisper: {e}")
            else:
                st.error("Whisper n'a pas pu être chargé pour cette langue.")
                # Log ajouté
                logger.error("Modèle Whisper non disponible.")
        
        # 2. Vosk pour les autres langues
        else:
            # Log ajouté
            logger.info(f"Utilisation de Vosk pour {lang_name}.")
            vosk_model = self.model_manager.download_vosk_model(lang_name)
            if vosk_model:
                # Log ajouté
                logger.info("Modèle Vosk disponible, démarrage de la reconnaissance.")
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
                        # Log ajouté
                        logger.info(f"Transcription Vosk réussie: '{transcription}'")
                except Exception as e:
                     st.warning(f"Erreur Vosk lors de la reconnaissance : {e}")
                     # Log ajouté
                     logger.warning(f"Erreur Vosk: {e}")
            
            # 3. Fallback Whisper si Vosk échoue ou est vide
            if not transcription.strip():
                # Log ajouté
                logger.warning(f"Vosk n'a pas produit de résultat pour {lang_name}.")
                st.warning(f"Vosk n'a pas pu transcrire ou n'est pas disponible pour {lang_name}. Tentative de fallback avec Whisper...")
                whisper_model = self.model_manager.get_whisper_model(model_size="tiny") # Utilise tiny pour la vitesse
                if whisper_model:
                    code = LANGUAGES_DATA[lang_name]["seamless_code"].strip("_")
                    # Log ajouté
                    logger.info(f"Code Fallback Whisper: {code}. Fichier à transcrire: {wav_path}")
                    try:
                        result = whisper_model.transcribe(wav_path, language=code)
                        transcription = result["text"]
                        # Log ajouté
                        logger.info(f"Transcription Fallback Whisper réussie: '{transcription}'")
                    except Exception as e:
                        # Amélioration du message d'erreur pour aider au diagnostic FFmpeg
                        if "WinError 2" in str(e):
                            error_msg = f"❌ Erreur Whisper Fallback: Le programme externe (probablement FFmpeg) est introuvable. Veuillez vérifier votre PATH. Détail: {e}"
                            st.error(error_msg)
                            logger.error(error_msg)
                        else:
                            st.error(f"Erreur Whisper Fallback: {e}")
                            # Log ajouté
                            logger.error(f"Erreur lors de la transcription Fallback Whisper: {e}")
                else:
                    # Log ajouté
                    logger.error("Modèle Whisper Fallback non disponible.")

        os.unlink(wav_path)
        final_transcription = transcription.strip()
        # Log ajouté
        logger.info(f"Transcription finale renvoyée: '{final_transcription}'")
        return final_transcription

    # --- Traduction ---
    def translate(self, text, src, tgt):
        # Log ajouté
        logger.info(f"Démarrage de la traduction. Source: {src}, Cible: {tgt}")
        logger.info(f"Texte à traduire: '{text}'")
        model, processor = self.model_manager.get_translator_pipeline()
        
        if not model or not processor:
            st.error("Modèle de traduction non disponible.")
            # Log ajouté
            logger.error("Modèle ou processeur SeamlessM4T non disponible.")
            return ""

        # Utilisez les codes Seamless pour la traduction
        src_seamless_code = LANGUAGES_DATA[src]["seamless_code"]
        tgt_seamless_code = LANGUAGES_DATA[tgt]["seamless_code"]
        # Log ajouté
        logger.info(f"Codes Seamless: {src_seamless_code} -> {tgt_seamless_code}")

        try:
            inputs = processor(text=text.strip(), src_lang=src_seamless_code, return_tensors="pt")
            # Log ajouté
            logger.debug(f"Inputs préparés: {inputs.keys()}")
            
            with torch.no_grad():
                output = model.generate(**inputs,
                                        tgt_lang=tgt_seamless_code,
                                        generate_speech=False,
                                        max_length=256)
            
            translated = processor.decode(output.sequences[0].tolist(), skip_special_tokens=True)
            # Nettoyage des tokens spéciaux Seamless (ex: __eng__)
            clean = re.sub(r'__[\w_]+__', '', translated).strip()
            # Logs ajoutés
            logger.info(f"Traduction brute: '{translated}'")
            logger.info(f"Traduction finale (nettoyée): '{clean}'")
            return clean
        except Exception as e:
            st.error(f"Erreur lors de la traduction SeamlessM4T: {e}")
            # Log ajouté
            logger.error(f"Erreur de traduction: {e}")
            return ""


    # --- Synthèse Vocale (TTS) ---
    def speak(self, text, lang_name):
        # Log ajouté
        logger.info(f"Démarrage de la synthèse vocale pour le texte: '{text[:30]}...' en {lang_name}")
        target_tts_code = LANGUAGES_DATA[lang_name]["tts_lang_code"]
        # Log ajouté
        logger.info(f"Code TTS utilisé: {target_tts_code}")

        if target_tts_code in ["JP", "ZH"]:
            self._speak_melo(text, target_tts_code)
        else:
            self._speak_mms(text, target_tts_code)

    def _speak_mms(self, text, lang_code):
        """ Synthèse vocale MMS-TTS utilisant VitsModel et AutoTokenizer. """
        model_id = MMS_TTS_MODELS[lang_code] 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Log ajouté
        logger.info(f"MMS-TTS: Chargement de {model_id} sur {device}")
        
        try:
            st.info(f"Génération audio pour {lang_code} avec {model_id}...")
            
            # Utilisation de la méthode from_pretrained pour charger le modèle directement
            model = VitsModel.from_pretrained(model_id).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            # Log ajouté
            logger.debug("Modèle et tokenizer MMS chargés.")

            inputs = tokenizer(text, return_tensors="pt").to(device)

            with torch.no_grad():
                waveform = model(**inputs).waveform

            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            sampling_rate = model.config.sampling_rate
            sf.write(tmp_wav, waveform.squeeze().cpu().numpy(), sampling_rate)
            # Log ajouté
            logger.info(f"Fichier audio MMS créé. Taux d'échantillonnage: {sampling_rate}")
            st.audio(tmp_wav, sample_rate=sampling_rate)
            os.unlink(tmp_wav)
            # Log ajouté
            logger.debug("Fichier audio MMS nettoyé.")

        except Exception as e:
            st.error(f"❌ Erreur lors de la synthèse vocale MMS : {e}")
            logger.error(f"Erreur MMS TTS: {e}")

    def _speak_melo(self, text, lang_code):
        """
        Synthèse vocale pour JA/ZH en appelant le script externe melo_tts_service.py via subprocess.
        """
        # Log ajouté
        logger.info(f"MeloTTS: Préparation du subprocess pour {lang_code}.")
        try:
            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            
            # Préparation des données JSON à passer au script externe
            input_data = {
                'text': text,
                'output_path': tmp_wav,
                'lang': lang_code
            }
            json_input = json.dumps(input_data)
            # Log ajouté
            logger.debug(f"MeloTTS Input JSON: {json_input[:100]}...")
            
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
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True 
            )
            
            if "Success" in result.stdout:
                # Log ajouté
                logger.info(f"MeloTTS Success. Fichier généré: {tmp_wav}")
                # MeloTTS utilise typiquement 22050 Hz.
                st.audio(tmp_wav, sample_rate=22050) 
            else:
                st.error(f"❌ Échec de la synthèse vocale MeloTTS. Erreur: {result.stderr}")
                # Log ajouté
                logger.error(f"MeloTTS Échec. Sortie: {result.stdout}. Erreur: {result.stderr}")
                
            os.unlink(tmp_wav)
            # Log ajouté
            logger.debug("Fichier audio MeloTTS nettoyé.")
            
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Erreur d'exécution de MeloTTS. Code de retour: {e.returncode}. Sortie d'erreur: {e.stderr}")
            logger.error(f"MeloTTS Subprocess Error (Code: {e.returncode}): {e.stderr}")
        except Exception as e:
            st.error(f"❌ Erreur générale lors de l'appel de MeloTTS: {e}")
            logger.error(f"MeloTTS Error: {e}")