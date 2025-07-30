"""
Classificatore AI per testi - Modulo principale
"""
import os
import sys
from typing import Tuple, Optional
from .model_utils import ModelManager
from ..data.training_data import ALL_TRAINING_DATA
from .config import CATEGORIES

class AITextClassifier:
    """
    Classificatore di testi per categorie AI ottimizzato per RTX 3060TI
    
    Categorie supportate:
    - AI generica
    - AI generativa  
    - Computer Vision
    - Robotica AI
    - Guida Autonoma
    - Data Science
    - AI Medica
    - ALTRO (per testi non AI)
    """
    
    def __init__(self, auto_train: bool = True):
        """
        Inizializza il classificatore
        
        Args:
            auto_train: Se True, addestra automaticamente il modello se non esiste
        """
        self.model_manager = ModelManager()
        self.is_trained = False
        
        # Carica o crea il modello
        model_exists = self.model_manager.load_or_create_model()
        
        if not model_exists and auto_train:
            print("Modello non trovato. Avvio training automatico...")
            self.train()
        
        self.is_trained = model_exists or auto_train
    
    def classify(self, text: str, return_confidence: bool = False) -> str | Tuple[str, float]:
        """
        Classifica un testo in una delle categorie AI
        
        Args:
            text: Il testo da classificare
            return_confidence: Se True, restituisce anche il livello di confidenza
            
        Returns:
            La categoria predetta, opzionalmente con la confidenza
            
        Raises:
            ValueError: Se il testo è vuoto o il modello non è addestrato
        """
        if not text or not text.strip():
            raise ValueError("Il testo non può essere vuoto")
        
        if not self.is_trained:
            raise ValueError("Il modello non è stato addestrato. Chiamare train() prima di classify()")
        
        try:
            # Predizione
            predicted_class, confidence = self.model_manager.predict(text.strip())
            category = CATEGORIES[predicted_class]
            
            if return_confidence:
                return category, confidence
            else:
                return category
                
        except Exception as e:
            print(f"Errore durante la classificazione: {e}")
            # Fallback per errori
            return ("ALTRO", 0.0) if return_confidence else "ALTRO"
    
    def classify_batch(self, texts: list[str], return_confidence: bool = False) -> list:
        """
        Classifica una lista di testi
        
        Args:
            texts: Lista di testi da classificare
            return_confidence: Se True, include la confidenza nei risultati
            
        Returns:
            Lista delle categorie predette (opzionalmente con confidenze)
        """
        results = []
        for text in texts:
            try:
                result = self.classify(text, return_confidence)
                results.append(result)
            except Exception as e:
                print(f"Errore nel classificare '{text[:50]}...': {e}")
                fallback = ("ALTRO", 0.0) if return_confidence else "ALTRO"
                results.append(fallback)
        
        return results
    
    def train(self, custom_data: Optional[list] = None):
        """
        Addestra il modello
        
        Args:
            custom_data: Dati personalizzati nel formato [(testo, categoria_id), ...]
                        Se None, usa i dati predefiniti
        """
        training_data = custom_data if custom_data is not None else ALL_TRAINING_DATA
        
        print(f"Training con {len(training_data)} esempi...")
        print("Categorie:", {v: k for k, v in CATEGORIES.items()})
        
        try:
            self.model_manager.train_model(training_data)  
            self.is_trained = True
            print("Training completato con successo!")
            
        except Exception as e:
            print(f"Errore durante il training: {e}")
            self.is_trained = False
            raise e
    
    def get_categories(self) -> dict:
        """Restituisce il dizionario delle categorie disponibili"""
        return CATEGORIES.copy()
    
    def get_model_info(self) -> dict:
        """Restituisce informazioni sul modello e sull'hardware"""
        info = {
            "categories": self.get_categories(),
            "is_trained": self.is_trained,
            "device": str(self.model_manager.device),
            "memory_usage": self.model_manager.get_memory_usage()
        }
        return info
    
    def cleanup(self):
        """Pulisce la memoria GPU"""
        self.model_manager.cleanup()
        print("Memoria GPU pulita")

# Funzioni di utility per uso esterno
def quick_classify(text: str) -> str:
    """
    Funzione rapida per classificare un singolo testo
    
    Args:
        text: Testo da classificare
        
    Returns:
        Categoria predetta
    """
    classifier = AITextClassifier()
    return classifier.classify(text)

def classify_with_confidence(text: str) -> Tuple[str, float]:
    """
    Classifica un testo restituendo anche la confidenza
    
    Args:
        text: Testo da classificare
        
    Returns:
        Tupla (categoria, confidenza)
    """
    classifier = AITextClassifier()
    return classifier.classify(text, return_confidence=True)