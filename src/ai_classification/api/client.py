import requests
import json
from typing import List, Dict, Optional
import time

class AIClassificationClient:
    """Client per comunicare con il server di classificazione AI"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def is_server_healthy(self) -> bool:
        """Verifica se il server è attivo e il modello è caricato"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def predict(self, text: str) -> Optional[Dict]:
        """Fai una singola predizione"""
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json={"text": text},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Errore nella richiesta: {e}")
            return None
    
    def predict_batch(self, texts: List[str]) -> Optional[List[Dict]]:
        """Fai predizioni multiple (più efficiente per molti testi)"""
        try:
            response = requests.post(
                f"{self.base_url}/predict_batch",
                json=texts,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Errore nella richiesta batch: {e}")
            return None
    
    def get_category_name(self, prediction: int) -> str:
        """Converte il numero di categoria nel nome"""
        categories = {
            0: "Altro",
            1: "AI Generica", 
            2: "AI Generativa",
            3: "Computer Vision",
            4: "Robotica AI",
            5: "Guida Autonoma",
            6: "Data Science",
            7: "AI Medica"
        }
        return categories.get(prediction, "Sconosciuto")

# Funzioni di utilità per uso semplice
def classify_text(text: str) -> tuple[str, float]:
    """Funzione semplice per classificare un singolo testo"""
    client = AIClassificationClient()
    
    if not client.is_server_healthy():
        raise Exception("Server non disponibile. Assicurati che sia in esecuzione con: python server.py")
    
    result = client.predict(text)
    if result:
        return result['category'], result['confidence']
    else:
        raise Exception("Errore nella classificazione")

def classify_texts(texts: List[str]) -> List[tuple[str, float]]:
    """Funzione semplice per classificare più testi"""
    client = AIClassificationClient()
    
    if not client.is_server_healthy():
        raise Exception("Server non disponibile. Assicurati che sia in esecuzione con: python server.py")
    
    results = client.predict_batch(texts)
    if results:
        return [(r['category'], r['confidence']) for r in results]
    else:
        raise Exception("Errore nella classificazione batch")

# Esempio di utilizzo
if __name__ == "__main__":
    client = AIClassificationClient()
    
    # Verifica che il server sia attivo
    print("Verifica connessione al server...")
    if not client.is_server_healthy():
        print("❌ Server non disponibile!")
        print("Avvia il server con: python server.py")
        exit(1)
    
    print("✅ Server attivo e funzionante!")
    print("-" * 60)
    
    # Test singola predizione
    print("🧪 Test singola predizione:")
    text = "Reti neurali convoluzionali per il riconoscimento di immagini"
    result = client.predict(text)
    
    if result:
        print(f"Testo: {text}")
        print(f"Categoria: {result['category']}")
        print(f"Confidenza: {result['confidence']:.4f}")
        print("-" * 60)
    
    # Test predizioni multiple
    print("🚀 Test predizioni multiple:")
    texts = [
        "Algoritmi di machine learning per la classificazione",
        "Ricetta della carbonara tradizionale",
        "Generazione di immagini con Stable Diffusion",
        "Bracci robotici per l'industria automobilistica",
        "Veicoli autonomi con sensori LiDAR",
        "Analisi dei dati per business intelligence",
        "Diagnosi medica assistita da AI"
    ]
    
    start_time = time.time()
    results = client.predict_batch(texts)
    end_time = time.time()
    
    if results:
        print(f"⏱️  Predizioni completate in {end_time - start_time:.2f} secondi")
        print(f"📊 Velocità: {len(texts)/(end_time - start_time):.1f} predizioni/secondo")
        print()
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Testo: {result['text'][:50]}...")
            print(f"   Categoria: {result['category']}")
            print(f"   Confidenza: {result['confidence']:.4f}")
            print()
    
    # Test delle funzioni di utilità
    print("🔧 Test funzioni di utilità:")
    try:
        category, confidence = classify_text("Deep learning per computer vision")
        print(f"Categoria: {category}, Confidenza: {confidence:.4f}")
        
        categories_conf = classify_texts([
            "Generazione di testo con GPT",
            "Storia dell'arte italiana"
        ])
        for cat, conf in categories_conf:
            print(f"Categoria: {cat}, Confidenza: {conf:.4f}")
            
    except Exception as e:
        print(f"Errore: {e}")
