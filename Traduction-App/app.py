import streamlit as st
from transformers import pipeline
import speech_recognition as sr
from gtts import gTTS
import io

# Dictionnaire pour mapper les noms de langues aux codes
LANGUAGES_TO_CODES = {
    "Français": "fra",
    "Anglais": "eng",
    "Japonais": "jpn",
}

# Mapping pour speech_recognition (BCP-47)
SPEECH_LANGUAGES_MAPPING = {
    "Français": "fr-FR",
    "Anglais": "en-US",
    "Japonais": "ja-JP",
}

# Mapping spécifique pour gTTS (ISO 639-1 simple)
GTTS_LANG_CODES = {
    "Français": "fr",
    "Anglais": "en",
    "Japonais": "ja",
}

# Mapping pour les modèles Helsinki-NLP
# Noms de modèles exacts pour les paires de langues
MODEL_MAPPING = {
    ("Japonais", "Anglais"): "Helsinki-NLP/opus-mt-ja-en",
    ("Anglais", "Japonais"): "Helsinki-NLP/opus-mt-en-jap",
    ("Français", "Anglais"): "Helsinki-NLP/opus-mt-fr-en",
    ("Anglais", "Français"): "Helsinki-NLP/opus-mt-en-fr",
}

# Fonction pour charger le modèle avec la mise en cache.
@st.cache_resource
def get_translator(src_lang, tgt_lang):
    """
    Initialise et met en cache la pipeline de traduction.
    """
    model_name = MODEL_MAPPING.get((src_lang, tgt_lang))
    if model_name:
        try:
            return pipeline(
                "translation",
                model=model_name
            )
        except Exception:
            return pipeline(
                "translation",
                model=model_name,
                device="cpu"
            )
    else:
        st.error(f"Modèle de traduction non disponible pour la paire {src_lang} -> {tgt_lang}")
        return None

### Fonctions d'aide pour l'audio
def transcribe_audio(lang_code_sr):
    """
    Écoute le microphone et transcrit l'audio en texte.
    Les paramètres sont ajustés pour permettre des pauses plus longues.
    """
    r = sr.Recognizer()
    
    # Ajuster les paramètres de sensibilité
    r.pause_threshold = 2.0  # Attendre 2 secondes de silence avant de considérer que la phrase est finie
    r.non_speaking_duration = 1.0  # Durée de non-parole avant de commencer l'écoute
    r.energy_threshold = 300  # Seuil d'énergie (peut être ajusté selon l'environnement)
    
    with sr.Microphone() as source:
        st.info("Ajustement du bruit ambiant... Veuillez patienter.")
        r.adjust_for_ambient_noise(source, duration=2.0)  # Ajustement plus long
        
        # Affichage des paramètres pour debug
        # OPTIONNEL ---  st.info(f"Paramètres d'écoute - Pause threshold: {r.pause_threshold}s, Energy threshold: {r.energy_threshold}") ----
        st.info("Parlez maintenant... L'écoute s'arrêtera après 2 secondes de silence.")
        
        try:
            # Configuration optimisée pour des phrases plus longues
            audio = r.listen(
                source, 
                timeout=10,  # Temps d'attente avant le premier son (réduit pour éviter les timeouts)
                phrase_time_limit=None  # Pas de limite de durée de phrase
            )
            
            st.info("Transcription en cours...")
            transcribed_text = r.recognize_google(audio, language=lang_code_sr)
            return transcribed_text
            
        except sr.WaitTimeoutError:
            st.warning("Aucun son détecté pendant 10 secondes. Veuillez commencer à parler plus rapidement.")
            return ""
        except sr.UnknownValueError:
            st.warning("Impossible de comprendre l'audio. Veuillez réessayer en parlant plus clairement.")
            return ""
        except sr.RequestError as e:
            st.error(f"Erreur de l'API de transcription (vérifiez votre connexion internet) : {e}")
            return ""

def speak_text(text, lang_code_gtts):
    """
    Convertit le texte en parole et joue l'audio.
    Utilise lang_code_gtts pour gTTS.
    """
    try:
        if not text.strip():
            st.warning("Aucun texte à lire.")
            return

        tts = gTTS(text=text, lang=lang_code_gtts)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        st.audio(fp, format='audio/mp3')
    except Exception as e:
        st.error(f"Erreur lors de la génération de l'audio avec gTTS. Vérifiez la connexion internet et le code de langue ({lang_code_gtts}) : {e}")

### Interface de l'Application Streamlit
st.title("Interprète Multilingue")
st.markdown("---")

# Widgets pour la sélection des langues
source_lang_name = st.selectbox(
    "Choisissez la langue de votre interlocuteur (source) :",
    list(LANGUAGES_TO_CODES.keys()),
    index=0
)
target_lang_name = st.selectbox(
    "Choisissez votre langue (cible) :",
    list(LANGUAGES_TO_CODES.keys()),
    index=1
)
st.markdown("---")

# Codes pour SpeechRecognition et gTTS
source_code_sr = SPEECH_LANGUAGES_MAPPING.get(source_lang_name)
target_code_sr = SPEECH_LANGUAGES_MAPPING.get(target_lang_name)
source_code_gtts = GTTS_LANG_CODES.get(source_lang_name)
target_code_gtts = GTTS_LANG_CODES.get(target_lang_name)

# Création de la paire de langues
lang_pair = (source_lang_name, target_lang_name)
rev_lang_pair = (target_lang_name, source_lang_name)

if source_lang_name == target_lang_name:
    st.warning("Veuillez sélectionner des langues source et cible différentes.")
elif lang_pair not in MODEL_MAPPING or rev_lang_pair not in MODEL_MAPPING:
    st.warning(f"La traduction n'est pas prise en charge pour la paire de langues **{source_lang_name} ↔️ {target_lang_name}**.")
else:
    st.subheader("1. Entendre ce que l'interlocuteur dit")
    if st.button("Démarrer l'écoute", key="start_listening"):
        with st.spinner('Écoute en cours du microphone...'):
            transcribed_text = transcribe_audio(source_code_sr)
            if transcribed_text:
                st.session_state.last_transcription = transcribed_text
                st.success("Transcription réussie !")
                st.write(f"**Texte de l'interlocuteur :** {st.session_state.last_transcription}")
            else:
                st.session_state.last_transcription = ""
                st.error("Échec de la transcription.")

    if 'last_transcription' in st.session_state and st.session_state.last_transcription:
        st.markdown("---")
        st.subheader("2. Comprendre le message")
        with st.spinner('Traduction du message...'):
            translator = get_translator(source_lang_name, target_lang_name)
            if translator:
                translated_text_for_user = translator(st.session_state.last_transcription)[0]['translation_text']
                st.session_state.translated_text_for_user = translated_text_for_user
                st.info(f"**Traduction pour vous :** {st.session_state.translated_text_for_user}")
            else:
                st.session_state.translated_text_for_user = ""
    else:
        st.session_state.translated_text_for_user = ""

    st.markdown("---")
    st.subheader("3. Répondre et faire entendre votre message")
    user_response = st.text_area("Entrez votre réponse (dans votre langue) :", value=st.session_state.get('user_response', ""), height=100)
    st.session_state.user_response = user_response

    if st.button("Traduire et parler à l'interlocuteur", key="speak_response"):
        if st.session_state.user_response:
            with st.spinner('Traduction et synthèse vocale en cours...'):
                translator = get_translator(target_lang_name, source_lang_name)
                if translator:
                    translated_response_for_interlocutor = translator(st.session_state.user_response)[0]['translation_text']
                    
                    st.write(f"**Texte à lire à haute voix pour l'interlocuteur :** {translated_response_for_interlocutor}")
                    
                    speak_text(translated_response_for_interlocutor, source_code_gtts)
                else:
                    st.error("Échec de la traduction.")
        else:
            st.warning("Veuillez entrer une réponse à traduire et à parler.")