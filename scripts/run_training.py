#!/usr/bin/env python3
"""
Script per avviare il training del modello AI
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_classifier import AITextClassifier
from src.ai_classification.data.training_data import ALL_TRAINING_DATA

def main():
    print("🚀 AVVIO TRAINING MODELLO AI")
    print("=" * 50)
    
    # Mostra statistiche dataset
    print(f"📊 Dataset: {len(ALL_TRAINING_DATA)} esempi totali")
    
    # Conta esempi per categoria
    category_counts = {}
    for text, label in ALL_TRAINING_DATA:
        category_counts[label] = category_counts.get(label, 0) + 1
    
    print("\n📈 Distribuzione categorie:")
    categories = {
        0: "ALTRO",
        1: "AI Generica", 
        2: "AI Generativa",
        3: "Computer Vision",
        4: "Robotica AI",
        5: "Guida Autonoma",
        6: "Data Science",
        7: "AI Medica"
    }
    
    for label, count in sorted(category_counts.items()):
        category_name = categories.get(label, f"Categoria {label}")
        print(f"  {category_name}: {count} esempi")
    
    print("\n🎯 Inizializzo classificatore...")
    
    try:
        # Crea classificatore senza auto-training
        classifier = AITextClassifier(auto_train=False)
        
        print("⚙️ Avvio training...")
        print("   - Utilizzo GPU se disponibile")
        print("   - Modello ottimizzato per RTX 3060TI")
        
        # Avvia training
        classifier.train()
        
        print("\n✅ TRAINING COMPLETATO CON SUCCESSO!")
        
        # Test rapido
        print("\n🧪 Test rapido del modello:")
        test_texts = [
            "Finanziamento per progetti di computer vision",
            "Bando per startup di intelligenza artificiale generativa", 
            "Contributi per robotica collaborativa",
            "Ricetta della pasta alla carbonara"
        ]
        
        for text in test_texts:
            try:
                category, confidence = classifier.classify(text, return_confidence=True)
                print(f"  '{text[:40]}...' → {category} ({confidence:.1%})")
            except Exception as e:
                print(f"  Errore nel test: {e}")
        
        # Mostra info finali
        print(f"\n📋 Info modello:")
        info = classifier.get_model_info()
        print(f"  Device: {info['device']}")
        print(f"  Memoria GPU: {info['memory_usage']}")
        
        classifier.cleanup()
        print("\n🎉 Training completato e modello salvato!")
        
    except Exception as e:
        print(f"\n❌ ERRORE DURANTE IL TRAINING: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
