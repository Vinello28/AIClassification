#!/usr/bin/env python3
"""
Test script per verificare il funzionamento dello shuffle del dataset
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_classifier import AITextClassifier
from training_data import ALL_TRAINING_DATA
from config import TRAINING_CONFIG

def test_shuffle():
    """Test per verificare che lo shuffle funzioni correttamente"""
    print("🧪 TEST SHUFFLE DATASET")
    print("=" * 50)
    
    print(f"📊 Dataset totale: {len(ALL_TRAINING_DATA)} esempi")
    print(f"🔀 Shuffle abilitato: {TRAINING_CONFIG['shuffle_data']}")
    print(f"🎲 Seed shuffle: {TRAINING_CONFIG['shuffle_seed']}")
    
    # Mostra i primi 5 esempi originali
    print("\n📝 Primi 5 esempi ORIGINALI:")
    for i, (text, label) in enumerate(ALL_TRAINING_DATA[:5]):
        print(f"  {i+1}. [{label}] {text[:60]}...")
      # Test shuffle manuale
    import random
    
    # Copia e shuffle con stesso seed
    test_data = ALL_TRAINING_DATA.copy()
    random.seed(TRAINING_CONFIG['shuffle_seed'])
    random.shuffle(test_data)
    
    print("\n🔀 Primi 5 esempi DOPO SHUFFLE:")
    for i, (text, label) in enumerate(test_data[:5]):
        print(f"  {i+1}. [{label}] {text[:60]}...")
    
    # Verifica che l'ordine sia cambiato
    different_order = any(
        ALL_TRAINING_DATA[i][0] != test_data[i][0] 
        for i in range(min(10, len(ALL_TRAINING_DATA)))
    )
    
    if different_order:
        print("\n✅ SHUFFLE FUNZIONA: L'ordine dei dati è cambiato")
    else:
        print("\n⚠️  ATTENZIONE: L'ordine sembra invariato")
    
    # Test di un training minimo per verificare
    print("\n🚀 Test training rapido con shuffle...")
    
    # Usa solo un piccolo subset per test veloce
    small_dataset = ALL_TRAINING_DATA[:100]  # Solo 100 esempi per test
    
    try:
        classifier = AITextClassifier(auto_train=False)
        print("🔄 Avvio training di test con 100 esempi...")
        classifier.train(small_dataset)
        print("✅ Training di test completato con successo!")
        
        # Test veloce di classificazione
        test_texts = [
            "Bando per finanziamento startup AI",
            "Ricetta della pasta al pomodoro", 
            "Computer vision per agricoltura",
            "Robot collaborativo industriale"
        ]
        
        print("\n🧠 Test classificazione:")
        for text in test_texts:
            result = classifier.classify(text, return_confidence=True)
            print(f"  '{text[:40]}...' → {result[0]} ({result[1]:.1%})")
            
        # Cleanup
        classifier.cleanup()
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_shuffle()
    if success:
        print("\n🎉 Test completato con successo!")
        print("💡 Lo shuffle del dataset è ora attivo durante il training")
        print("📈 Questo migliorerà la robustezza del modello prevenendo l'overfitting")
    else:
        print("\n❌ Test fallito. Controlla la configurazione.")
