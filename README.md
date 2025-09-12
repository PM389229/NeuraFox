🧠 Interprète Multilingue (100% Offline)
Cette application Streamlit est un interprète multilingue capable de transcrire, traduire et synthétiser de la voix de manière entièrement hors ligne. Elle a été conçue pour fonctionner sans connexion internet, en utilisant des modèles locaux.

🚀 Fonctionnalités
Transcription audio intelligente : Détection automatique du silence pour arrêter l'enregistrement.

Traduction hors ligne : Utilise un modèle SeamlessM4T pour la traduction entre le français, l'anglais et le japonais.

Synthèse vocale (TTS) : Génère un fichier audio pour la traduction, en utilisant des modèles spécifiques pour chaque langue.

Mode conversation : Permet de transcrire votre interlocuteur, de traduire sa phrase et de la prononcer, puis de faire l'inverse pour votre réponse.

🛠️ Modèles et dépendances
L'application repose sur plusieurs modèles et bibliothèques Python.

Transcription :

Vosk pour le français et l'anglais.

Whisper pour le japonais.

Traduction : SeamlessM4T de Hugging Face.

Synthèse vocale :

coqui-tts pour le français et l'anglais.

MeloTTS pour le japonais.

En raison de la taille importante des modèles, ils ne sont pas inclus dans ce dépôt GitHub. Vous devez les télécharger et les placer dans les chemins spécifiés dans le code.

⚙️ Structure des dossiers
Pour que l'application fonctionne correctement, vos modèles doivent être placés aux chemins indiqués dans le fichier app.py. Voici la structure de dossier attendue :

Traduction-App/
├── app.py
├── generate_tts_jp.py
├── README.md
├── .gitignore
├── hf-seamless-m4t-medium/  # Modèle de traduction SeamlessM4T
├── vosk-model-en/          # Modèle Vosk Anglais
├── vosk-model-fr/          # Modèle Vosk Français
├── models/
│   ├── tts/
│       └── melotts-japanese/  # Modèle TTS Japonais
└── venv_py311/             # Environnement virtuel de l'application Streamlit
└── melotts_env/            # Environnement virtuel dédié à MeloTTS

📝 Installation et Lancement
Suivez ces étapes pour installer et lancer l'application.

Cloner le dépôt GitHub.

git clone <URL_DE_VOTRE_DÉPÔT>
cd Traduction-App

Créer les deux environnements virtuels.

L'un pour Streamlit et les dépendances principales (venv_py311).

L'autre spécifiquement pour MeloTTS (melotts_env).

Installer les dépendances.

Pour venv_py311 : installez streamlit, transformers, torch, whisper, vosk, coqui-tts.

Pour melotts_env : installez melo-tts et transformers.

Télécharger les modèles locaux.

Téléchargez les modèles SeamlessM4T, Vosk, Whisper et MeloTTS et placez-les dans les dossiers correspondants.

Lancer l'application.

Activez l'environnement de l'application Streamlit.

Exécutez l'application avec la commande streamlit run app.py.

source venv_py311/bin/activate
streamlit run app.py

Utiliser l'application.

L'interface s'ouvrira dans votre navigateur.

Sélectionnez les langues, cliquez sur "Démarrer écoute intelligente" et suivez les étapes.

🤝 Contribution
Si vous souhaitez contribuer, n'hésitez pas à ouvrir une issue ou à soumettre une pull request.