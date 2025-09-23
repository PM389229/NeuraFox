






                    🧠 Interprète Multilingue Hors Ligne

On a créé ce projet pour pouvoir traduire et synthétiser des conversations, même sans connexion Internet. 


                    🌟 Ce que l'app peut faire


Transcription intelligente : L'app écoute ce que tu dis et le convertit en texte. 

Pour le français et l'anglais, elle utilise Vosk, et pour le japonais, on a opté pour Whisper d'OpenAI.

Détection de silence : Pas besoin d'appuyer sur un bouton pour arrêter l'enregistrement. L'app est faite pour s'arrêter seule après 2 secondes de silence.

Traduction hors ligne : Elle utilise le modèle SeamlessM4T de Meta pour traduire les phrases sans avoir besoin d'être connectée à Internet.


Synthèse vocale (TTS) : Une fois le texte traduit, l'app le transforme en voix. On a utilisé plusieurs scripts pour que ça sonne le plus naturel possible, peu importe la langue.





                 🛠️ L'installation : Les deux environnements virtuels



Pour que tout fonctionne comme sur des roulettes, il te faudra deux environnements virtuels. C'est la meilleure façon de gérer toutes les dépendances sans que tout ne s'emmêle.





1. D'abord, on clone le projet
Ouvre le terminal et tape ces commandes pour récupérer le code :

git clone [https://github.com/PM389229/NeuraFox.git]
cd votre-projet






2. Le premier environnement (principal)
On va l'appeler venv. Il contient la majorité des dépendances pour l'interface Streamlit et la reconnaissance vocale.

python -m venv venv
source venv/bin/activate  # Sur Mac/Linux
# venv\Scripts\activate  # Sur Windows
pip install -r requirements.txt

Note : N'oublie pas de vérifier que ton fichier requirements.txt est à jour et contient toutes les librairies, comme streamlit, transformers, torch, whisper, etc.





3. Le second environnement (spécial TTS japonais)

Ce modèle de synthèse vocale japonaise a des dépendances très spécifiques (qui rentraient en conflit avec le premier environnement virtuel , il nécessite des versions précises de torch et transformers ), d'où la nécessité d'un environnement séparé. On l'appelle  melotts_env.

python -m venv melotts_env
source melotts_env/bin/activate # Sur Mac/Linux
# melotts_env\Scripts\activate  # Sur Windows
pip install TTS[ja]





4. Télécharger les modèles



Vu que ces modèles sont très lourds, on ne les a pas mis dans le dépôt.
Il faudra les télécharger manuellement et les placer dans les bons dossiers. Voici les instructions précises pour chaque cas.



                       🎙️ Modèles de reconnaissance vocale (Vosk)

Ces modèles sont essentiels pour la transcription du français et de l'anglais. 
Ils sont légers et très performants pour l'écoute en temps réel.

Va sur le site officiel de Vosk : https://alphacephei.com/vosk/models

Cherche les modèles "French small model" et "English small model".

Télécharge les deux archives .zip correspondantes.

Une fois téléchargées, décompresse-les. Crée deux nouveaux dossiers à la racine de ton projet : vosk-model-fr et vosk-model-en. Place les fichiers extraits dans ces dossiers.

                   🎙️ Modèle de reconnaissance vocale japonais (Whisper)

Rien à faire ici ! Le modèle Whisper d'OpenAI est très pratique car il se télécharge automatiquement la première fois que tu lances un script qui l'utilise. La librairie est déjà incluse dans ton requirements.txt donc l'installation se fait en même temps que le reste des dépendances. 

🌐 Modèle de traduction (SeamlessM4T de Meta)

Ce modèle permet de traduire sans connexion Internet. Il est très volumineux, mais il est au cœur du projet.

Rends-toi sur la page Hugging Face du modèle : https://huggingface.co/meta-llama/seamless-m4t-medium

Dans l'onglet "Files and versions", clique sur "Download all files" pour télécharger le dossier complet. C'est une archive .zip ou tu peux utiliser git lfs si tu es familier avec.

Une fois téléchargé, renomme le dossier en hf-seamless-m4t-medium et place-le à la racine de ton projet.



                    🗣️ Modèles de synthèse vocale (TTS)
Ici, on a une petite particularité. On utilise la librairie melo.api.TTS pour générer la voix. Elle est très pratique, car elle s'occupe toute seule de télécharger les modèles la première fois que tu lances un script qui l'utilise.

TTS pour l'anglais, le français et le japonais :
Rien à faire ici ! La librairie melo.api.TTS est conçue pour gérer le téléchargement et la mise en cache des modèles automatiquement. La première fois que tu exécuteras un script utilisant un modèle pour l'une de ces langues, la librairie vérifiera si le modèle est présent. S'il n'est pas dans son dossier de cache par défaut, elle le téléchargera pour toi. C'est la raison pour laquelle il n'y a pas d'étapes de téléchargement manuel pour ces modèles dans ce guide.







                       🏃 Comment lancer l'application

Une fois que tous les modèles sont en place, tu peux lancer l'interface utilisateur.

streamlit run app.py

L'application va se lancer automatiquement dans ton navigateur.

☁️ Pour la mettre en ligne (Kaggle ou Colab)
Ça sera la prochaine étape. Quand tu seras prêt, il faudra juste uploader tous les fichiers et installer les dépendances et les modèles directement sur la plateforme.

🤝 Participer au projet
Si tu as des idées pour améliorer l'app, n'hésite pas ! Les pull requests sont les bienvenues.