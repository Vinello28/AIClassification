"""
Script di debug per testare il classificatore
"""
from ai_classifier import AITextClassifier
import torch

def test_classifier_detailed():
    print("=== Test Dettagliato del Classificatore ===")
    
    # Inizializza il classificatore
    classifier = AITextClassifier(auto_train=False)
    
    # Test specifici
    test_cases = [
        ("GPT-4 è un modello generativo", "AI generativa"),
        ("Riconoscimento oggetti in immagini", "Computer Vision"),
        ("Robot industriale per produzione", "Robotica AI"),
        ("Auto a guida autonoma", "Guida Autonoma"),
        ("Analisi big data", "Data Science"),
        ("Diagnosi medica con AI", "AI Medica"),
        ("Intelligenza artificiale generale", "AI generica"),
        ("Ricetta della carbonara", "ALTRO"),
    ]
    
    print("Predizioni del modello:")
    for i, (text, expected) in enumerate(test_cases):
        try:
            # Predizione dettagliata
            result = classifier.model_manager.predict(text)
            predicted_id, confidence = result
            predicted_category = classifier.get_categories()[predicted_id]
            
            print(f"\n{i+1}. Testo: '{text}'")
            print(f"   Attesa: {expected}")
            print(f"   Predetta: {predicted_category} (ID: {predicted_id})")
            print(f"   Confidenza: {confidence:.4f}")
            print(f"   Corretto: {'✓' if predicted_category == expected else '✗'}")
            
        except Exception as e:
            print(f"Errore su '{text}': {e}")
    
    # Test delle probabilità raw
    print("\n=== Test Raw Predictions ===")
    test_text = "GPT-4 genera testo automaticamente"
    try:
        # Tokenizzazione
        inputs = classifier.model_manager.tokenizer(
            test_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(classifier.model_manager.device)
        
        # Predizione
        with torch.no_grad():
            outputs = classifier.model_manager.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
        print(f"Testo: '{test_text}'")
        print("Probabilità per categoria:")
        categories = classifier.get_categories()
        for i, (cat_id, cat_name) in enumerate(categories.items()):
            prob = probabilities[0][i].item()
            print(f"  {cat_name}: {prob:.4f}")
            
    except Exception as e:
        print(f"Errore nel test raw: {e}")

if __name__ == "__main__":
    test_classifier_detailed()
