"""
Test unitari per il classificatore AI
"""
import unittest
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai_classification.core.classifier import AITextClassifier, quick_classify
from src.ai_classification.core.config import CATEGORIES

class TestAIClassifier(unittest.TestCase):
    """Test per la classe AITextClassifier"""
    
    @classmethod
    def setUpClass(cls):
        """Setup per tutti i test"""
        print("Inizializzazione classificatore per i test...")
        cls.classifier = AITextClassifier()
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup dopo tutti i test"""
        cls.classifier.cleanup()
    
    def test_initialization(self):
        """Testa l'inizializzazione del classificatore"""
        self.assertIsNotNone(self.classifier)
        self.assertTrue(hasattr(self.classifier, 'model_manager'))
    
    def test_categories_structure(self):
        """Testa la struttura delle categorie"""
        categories = self.classifier.get_categories()
        
        # Controlla che ci siano 8 categorie
        self.assertEqual(len(categories), 8)
        
        # Controlla che ALTRO sia presente
        self.assertIn("ALTRO", categories.values())
        
        # Controlla categorie specifiche
        expected_categories = [
            "ALTRO", "AI generica", "AI generativa", "Computer Vision",
            "Robotica AI", "Guida Autonoma", "Data Science", "AI Medica"
        ]
        
        for expected in expected_categories:
            self.assertIn(expected, categories.values())
    
    def test_empty_text_handling(self):
        """Testa la gestione di testi vuoti"""
        with self.assertRaises(ValueError):
            self.classifier.classify("")
        
        with self.assertRaises(ValueError):
            self.classifier.classify("   ")
        
        with self.assertRaises(ValueError):
            self.classifier.classify(None)
    
    def test_ai_generativa_classification(self):
        """Testa la classificazione di testi AI generativa"""
        test_cases = [
            "GPT-4 è un modello di linguaggio generativo",
            "ChatGPT per conversazioni intelligenti",
            "DALL-E genera immagini artistiche",
            "Stable Diffusion per la creatività AI"
        ]
        
        for text in test_cases:
            result = self.classifier.classify(text)
            # Dovrebbe classificare come AI generativa o almeno come AI generica
            self.assertIn(result, ["AI generativa", "AI generica"])
    
    def test_computer_vision_classification(self):
        """Testa la classificazione di testi Computer Vision"""
        test_cases = [
            "Riconoscimento facciale con deep learning",
            "Object detection tramite YOLO",
            "Segmentazione semantica di immagini",
            "OCR per il riconoscimento del testo"
        ]
        
        for text in test_cases:
            result = self.classifier.classify(text)
            # Dovrebbe classificare come Computer Vision o AI generica
            self.assertIn(result, ["Computer Vision", "AI generica"])
    
    def test_robotica_classification(self):
        """Testa la classificazione di testi Robotica AI"""
        test_cases = [
            "Robot industriali con controllo intelligente",
            "Bracci robotici per assemblaggio automatico",
            "Robot domestici con AI conversazionale"
        ]
        
        for text in test_cases:
            result = self.classifier.classify(text)
            self.assertIn(result, ["Robotica AI", "AI generica"])
    
    def test_guida_autonoma_classification(self):
        """Testa la classificazione di testi Guida Autonoma"""
        test_cases = [
            "Tesla Autopilot per la guida automatica",
            "Veicoli a guida autonoma con LiDAR",
            "Self-driving cars del futuro"
        ]
        
        for text in test_cases:
            result = self.classifier.classify(text)
            self.assertIn(result, ["Guida Autonoma", "AI generica"])
    
    def test_data_science_classification(self):
        """Testa la classificazione di testi Data Science"""
        test_cases = [
            "Big Data analytics con Apache Spark",
            "Machine learning per business intelligence",
            "Analisi predittiva su dataset massivi"
        ]
        
        for text in test_cases:
            result = self.classifier.classify(text)
            self.assertIn(result, ["Data Science", "AI generica"])
    
    def test_ai_medica_classification(self):
        """Testa la classificazione di testi AI Medica"""
        test_cases = [
            "Diagnosi medica assistita da intelligenza artificiale",
            "Analisi radiologiche con deep learning",
            "Drug discovery accelerato dall'AI"
        ]
        
        for text in test_cases:
            result = self.classifier.classify(text)
            self.assertIn(result, ["AI Medica", "AI generica"])
    
    def test_altro_classification(self):
        """Testa la classificazione di testi non-AI"""
        test_cases = [
            "La ricetta della pasta alla carbonara",
            "Storia dell'Impero Romano",
            "Calcio: risultati della Serie A",
            "Coltivazione di pomodori in giardino"
        ]
        
        for text in test_cases:
            result = self.classifier.classify(text)
            self.assertEqual(result, "ALTRO")
    
    def test_confidence_score(self):
        """Testa che la confidenza sia nel range corretto"""
        text = "GPT-4 è un modello generativo avanzato"
        category, confidence = self.classifier.classify(text, return_confidence=True)
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_batch_classification(self):
        """Testa la classificazione in batch"""
        texts = [
            "Computer vision per riconoscimento oggetti",
            "La cucina italiana tradizionale",
            "Robot industriali intelligenti"
        ]
        
        results = self.classifier.classify_batch(texts)
        
        self.assertEqual(len(results), len(texts))
        self.assertIn(results[0], ["Computer Vision", "AI generica"])
        self.assertEqual(results[1], "ALTRO")
        self.assertIn(results[2], ["Robotica AI", "AI generica"])
    
    def test_model_info(self):
        """Testa le informazioni del modello"""
        info = self.classifier.get_model_info()
        
        self.assertIn('categories', info)
        self.assertIn('is_trained', info)
        self.assertIn('device', info)
        self.assertIn('memory_usage', info)
        
        self.assertIsInstance(info['categories'], dict)
        self.assertIsInstance(info['is_trained'], bool)

class TestQuickFunctions(unittest.TestCase):
    """Test per le funzioni di utility rapide"""
    
    def test_quick_classify(self):
        """Testa la funzione quick_classify"""
        result = quick_classify("GPT-4 modello generativo")
        self.assertIsInstance(result, str)
        self.assertIn(result, CATEGORIES.values())
    
    def test_quick_classify_non_ai(self):
        """Testa quick_classify con testo non-AI"""
        result = quick_classify("Ricetta pasta al pomodoro")
        self.assertEqual(result, "ALTRO")

def run_performance_test():
    """Test delle performance (opzionale)"""
    import time
    
    print("\n=== Test Performance ===")
    classifier = AITextClassifier()
    
    # Test velocità classificazione singola
    text = "Modelli di machine learning per l'analisi predittiva"
    
    start_time = time.time()
    for _ in range(10):
        classifier.classify(text)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    print(f"Tempo medio classificazione: {avg_time:.3f} secondi")
    
    # Test classificazione batch
    batch_texts = [
        "Computer vision algorithms",
        "Pasta recipe ingredients", 
        "Autonomous driving sensors",
        "Medical AI diagnostics",
        "Robotics automation"
    ] * 4  # 20 testi totali
    
    start_time = time.time()
    results = classifier.classify_batch(batch_texts)
    end_time = time.time()
    
    batch_time = end_time - start_time
    print(f"Tempo classificazione batch (20 testi): {batch_time:.3f} secondi")
    print(f"Tempo per testo in batch: {batch_time/20:.3f} secondi")
    
    classifier.cleanup()

if __name__ == "__main__":
    # Esegue i test unitari
    print("Avvio test unitari...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Esegue test performance se richiesto
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--performance":
        run_performance_test()
