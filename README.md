




                               
                                Interprète Multilingue Hors Ligne
Nous avons créé ce projet pour pouvoir traduire et synthétiser des conversations, même sans connexion Internet.

                                    Ce que l'app peut faire

Transcription intelligente : L'app écoute ce que tu dis et le convertit en texte.

Pour le français et l'anglais, elle utilise Vosk.Pour le japonais et le mandarin, on a opté pour Whisper d'OpenAI.

Détection de silence : Pas besoin d'appuyer sur un bouton pour arrêter l'enregistrement. L'app est faite pour s'arrêter seule après 2 secondes de silence.

Traduction hors ligne : Elle utilise le modèle SeamlessM4T de Meta pour traduire les phrases sans avoir besoin d'être connectée à Internet.

Synthèse vocale (TTS) : Une fois le texte traduit, l'app le transforme en voix. On utilise des modèles différents pour un rendu naturel :

MMS-TTS pour le français et l'anglais.

MeloTTS (via un script séparé) pour le japonais (JA) et le mandarin (ZH), gérant ainsi les dépendances complexes.




                        L'installation : 



1. D'abord, on clone le projet
Ouvre le terminal et tape ces commandes pour récupérer le code :

git clone [[https://github.com/PM389229/NeuraFox.git](https://github.com/PM389229/NeuraFox.git)]
cd votre-projet

2 Création environnement virtuel

A ce path pour fonctionnement correct : C:\Users\User\Neurafox\venv_unified
Et : .\venv_unified\Scripts\activate

3 Lancement interface Streamlit

streamlit run app.py




                                                  
                                                  INFOS et Besoins





--->téléchargez unidic_mecab et à mettre ici : C:\Users\User\Neurafox\unidic-mecab-2.1.2
--->Lien ;  https://clrd.ninjal.ac.jp/unidic_archive/cwj/2.1.2/unidic-mecab-2.1.2_bin.zip (à decompresser et mettre ou précisé au dessus)
----> UniDic est la base de données de vocabulaire utilisée par l'analyseur MeCab. Il est indispensable pour la synthèse vocale japonaise (MeloTTS), car il fournit les informations d'accent tonique (pitch accent) et de prononciation nécessaires pour générer une voix qui sonne naturelle. Sans lui, le moteur TTS ne peut pas segmenter et prononcer le japonais correctemen


🌐 Modèle de traduction (SeamlessM4T de Meta)

Ce modèle permet de traduire sans connexion Internet. Il est très volumineux, mais il est au cœur du projet.

Son téléchargement est automatique à la première utilisation si détection de son absence


🗣️ Modèles de synthèse vocale (TTS)

Tous les modèles sont comme Seamless téléchargés automatiquement à leur non-détection





