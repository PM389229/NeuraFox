from transformers import pipeline

# Définir le modèle de traduction et la paire de langues que vous voulez tester
source_lang = "Anglais"
target_lang = "Japonais"

# Le modèle Helsinki-NLP pour la traduction Anglais vers Japonais
model_name = "Helsinki-NLP/opus-mt-en-jap"

# Phrase à traduire
text_to_translate = "Do you like fruits?"

try:
    # Initialisation de la pipeline de traduction
    translator = pipeline("translation", model=model_name)

    # Effectuer la traduction
    translation_result = translator(text_to_translate)
    
    # Récupérer la traduction et l'afficher
    translated_text = translation_result[0]['translation_text']
    
    print(f"Modèle utilisé : {model_name}")
    print(f"Paire de langues : {source_lang} -> {target_lang}")
    print(f"Texte original : {text_to_translate}")
    print(f"Traduction : {translated_text}")

except Exception as e:
    print(f"Une erreur est survenue lors du chargement ou de la traduction : {e}")