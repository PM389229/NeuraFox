




                               
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





                      
                      
                      Configuration spécifique par OS


🔹 Windows


Chemin UniDic pour MeloTTS

Téléchargez UniDic 2.1.2 :
https://clrd.ninjal.ac.jp/unidic_archive/cwj/2.1.2/unidic-mecab-2.1.2_bin.zip

Décompressez dans :

C:\Users\User\Neurafox\unidic-mecab-2.1.2


Ce chemin doit correspondre à la variable MECAB_DIC_PATH dans melo_tts_service.py.

Activation de l’environnement

.\venv_unified\Scripts\activate




🔹 macOS

Installer MeCab et IPADIC

brew install mecab mecab-ipadic


Installer UniDic pour Python

pip install unidic
python -m unidic download


UniDic est la base de vocabulaire utilisée par MeCab pour générer une voix japonaise naturelle avec MeloTTS. Sur macOS, le chemin est géré automatiquement via unidic.DICDIR.




                          
                          Activation de l’environnement

source tts-mac-single/bin/activate

                         Lancer l’interface Streamlit
streamlit run app.py
 



                         Modèles utilisés

SeamlessM4T (Meta)

Traduction hors ligne.

Très volumineux, téléchargement automatique à la première utilisation si absent.

MMS-TTS

Pour français et anglais.

Téléchargement automatique si non détecté.

MeloTTS

Pour japonais (JA) et mandarin (ZH).

Nécessite MeCab + UniDic.




⚠️ Notes importantes

Assurez-vous que MeCab et UniDic soient installés avant d’utiliser MeloTTS pour JA/ZH.

MMS-TTS pour FR/EN fonctionne sans configuration supplémentaire.

Le chemin du dictionnaire sur Windows doit être codé dans melo_tts_service.py.

Sur macOS, le chemin est géré automatiquement via unidic.DICDIR.





