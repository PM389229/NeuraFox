# app.py

import streamlit as st
from translator_service import TranslatorService
from config import LANGUAGES_DATA, MAX_RECORDING_TIME, SILENCE_DURATION, SILENCE_THRESHOLD
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='LOG: %(message)s')

# =======================
# INITIALISATION
# =======================

# Initialisation du service (une seule fois)
if 'translator_service' not in st.session_state:
    st.session_state.translator_service = TranslatorService()

translator_service = st.session_state.translator_service

# =======================
# INTERFACE STREAMLIT
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
if st.button("✅ Valider et charger modèles", key="load_models_btn"):
    st.session_state.selected_pair = (input_lang, target_lang)
    st.info(f"Téléchargement/vérification des modèles pour **{input_lang}** et **{target_lang}**...")

    # Téléchargement/Vérification des modèles ASR (pour la langue d'entrée)
    translator_service.model_manager.download_vosk_model(input_lang)
    
    # Téléchargement/Vérification des modèles TTS (pour les deux langues)
    input_tts_code = LANGUAGES_DATA[input_lang]["tts_lang_code"]
    target_tts_code = LANGUAGES_DATA[target_lang]["tts_lang_code"]
    translator_service.model_manager.download_tts_model(input_tts_code)
    translator_service.model_manager.download_tts_model(target_tts_code)
    
    # Préchage du modèle de traduction SeamlessM4T
    translator_service.model_manager.get_translator_pipeline() 

    st.success("✅ Modèles téléchargés et prêts ! Vous pouvez commencer à parler.")
    st.rerun() # Utilisation de st.rerun()

st.markdown("---")

# === ÉTAPE 1: TRANSCRIPTION ===
if "selected_pair" in st.session_state:
    st.subheader("1. 🎤 Transcription avec détection de silence")
    
    with st.expander("ℹ️ Paramètres d'écoute"):
        st.write(f"• Durée maximale : **{MAX_RECORDING_TIME} s**")
        st.write(f"• Arrêt automatique : **{SILENCE_DURATION} s** de silence")
        st.write(f"• Seuil de silence : **{SILENCE_THRESHOLD}** (RMS)")

    if st.button("🎤 Démarrer écoute intelligente", key="record_btn"):
        with st.spinner(f"Enregistrement en cours en {st.session_state.selected_pair[0]}... (Parlez et attendez le silence pour l'arrêt automatique)"):
            # Appel de la méthode de service
            transcription = translator_service.transcribe(st.session_state.selected_pair[0])
            
            if transcription and transcription.strip():
                st.session_state.last_transcription = transcription
                st.success(f"✅ Transcription réussie ({st.session_state.selected_pair[0]})")
                st.write(f"**Texte transcrit :** `{transcription}`")
            else:
                st.error("❌ Aucun texte détecté ou transcription échouée.")
            st.rerun() # Utilisation de st.rerun()


# === ÉTAPE 2: TRADUCTION ===
if st.session_state.get("last_transcription"):
    st.markdown("---")
    st.subheader("2. 🌍 Traduction")
    
    src_lang, tgt_lang = st.session_state.selected_pair
    st.write(f"Traduction : **{src_lang} → {tgt_lang}**")

    if st.button("🌍 Traduire", key="translate_btn"):
        with st.spinner("Traduction en cours..."):
            # Appel de la méthode de service
            translated_text = translator_service.translate(
                st.session_state.last_transcription,
                src_lang,
                tgt_lang
            )
            
            if translated_text:
                st.session_state.translated = translated_text
                st.success("✅ Traduction réussie")
                st.write(f"**Traduction :** `{translated_text}`")
            else:
                st.error("❌ Échec de la traduction")
            st.rerun() # Utilisation de st.rerun()

# === ÉTAPE 3: SYNTHÈSE VOCALE ===
if st.session_state.get("translated"):
    st.markdown("---")
    st.subheader("3. 🔊 Lecture audio de la traduction")
    st.write(f"**Texte à prononcer :** `{st.session_state.translated}`")
    target_lang_name = st.session_state.selected_pair[1]
    
    if st.button("🔊 Lire traduction", key="speak_btn"):
        # Appel de la méthode de service unifiée
        translator_service.speak(st.session_state.translated, target_lang_name)

# === ÉTAPE 4: VOTRE RÉPONSE (Inverse) ===
st.markdown("---")
st.subheader("4. 📝 Votre réponse")
target_lang_name = st.session_state.get('selected_pair', ('Langue Cible', ''))[1]
st.info(f"Écrivez votre réponse en **{target_lang_name}**")

user_response = st.text_area("Tapez votre réponse ici...", key="user_response_text")

if st.button("🔁 Traduire et prononcer votre réponse", key="inverse_btn"):
    if user_response.strip():
        src_lang, tgt_lang = st.session_state.selected_pair
        
        # Traduction inverse: Cible -> Source
        with st.spinner(f"Traduction de {tgt_lang} vers {src_lang}..."):
            translated_response = translator_service.translate(
                user_response,
                tgt_lang,  # Source est la langue de la réponse (Cible initiale)
                src_lang   # Cible est la langue de l'interlocuteur (Source initiale)
            )
        
        if translated_response:
            st.session_state.translated_response = translated_response
            st.success(f"✅ Traduction réussie")
            st.write(f"**Réponse traduite en {src_lang} :** `{translated_response}`")
            
            # Lecture audio de la réponse traduite (dans la langue Source initiale)
            translator_service.speak(translated_response, src_lang)
        else:
            st.error("❌ Échec de la traduction de votre réponse.")
    else:
        st.warning("Veuillez taper votre réponse avant de continuer.")

# === DEBUG / INFO SESSION ===
st.markdown("---")
with st.expander("🔧 Informations de débogage et réinitialisation"):
    st.write("État de la session Streamlit :")
    st.json({k: v if k not in ['translator_service'] else '<<Service Object>>' for k, v in st.session_state.items()})

    # === RÉINITIALISATION ===
    if st.button("🔄 Réinitialiser la session", key="reset_btn"):
        for key in list(st.session_state.keys()):
            if key != 'translator_service': # Garder l'objet service déjà chargé
                del st.session_state[key]
        st.success("Session réinitialisée. Rechargez la page ou cliquez à nouveau sur 'Valider et charger modèles'.")
        st.rerun() # Utilisation de st.rerun()