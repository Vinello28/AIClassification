"""
Classificatore ibrido che combina AI e keyword matching
"""
from ai_classifier import AITextClassifier
from simple_classifier import SimpleAIClassifier
import warnings
warnings.filterwarnings("ignore")

class HybridAIClassifier:
    """
    Classificatore ibrido che combina deep learning e keyword matching
    """
    
    def __init__(self, use_ai_threshold=0.3):
        """
        Inizializza il classificatore ibrido
        
        Args:
            use_ai_threshold: Soglia di confidenza sotto la quale usare keyword matching
        """
        self.use_ai_threshold = use_ai_threshold
        self.simple_classifier = SimpleAIClassifier()
        
        # Prova a caricare il modello AI, se fallisce usa solo keyword
        try:
            self.ai_classifier = AITextClassifier()
            self.has_ai_model = True
            print("✅ Modello AI caricato con successo")
        except Exception as e:
            print(f"⚠️  Modello AI non disponibile: {e}")
            print("🔧 Usando solo classificatore keyword-based")
            self.ai_classifier = None
            self.has_ai_model = False
    
    def classify(self, text, return_confidence=False):
        """
        Classifica un testo usando approccio ibrido
        
        Args:
            text: Testo da classificare
            return_confidence: Se restituire anche la confidenza
            
        Returns:
            Categoria (e confidenza se richiesta)
        """
        # Se non abbiamo il modello AI, usa solo keyword
        if not self.has_ai_model:
            return self.simple_classifier.classify(text, return_confidence)
        
        try:
            # Prova prima con il modello AI
            ai_category, ai_confidence = self.ai_classifier.classify(text, return_confidence=True)
            
            # Se la confidenza AI è alta, usa il risultato AI
            if ai_confidence >= self.use_ai_threshold:
                if return_confidence:
                    return ai_category, ai_confidence
                else:
                    return ai_category
            
            # Altrimenti, usa keyword matching
            keyword_category, keyword_confidence = self.simple_classifier.classify(text, return_confidence=True)
            
            # Scegli il metodo con confidenza più alta
            if keyword_confidence > ai_confidence:
                final_category = keyword_category
                final_confidence = keyword_confidence
                method_used = "keyword"
            else:
                final_category = ai_category
                final_confidence = ai_confidence
                method_used = "ai"
            
            # Stampa debug info
            if final_confidence < 0.5:
                print(f"🤔 Bassa confidenza per '{text[:30]}...': {final_category} ({final_confidence:.1%}, metodo: {method_used})")
            
            if return_confidence:
                return final_category, final_confidence
            else:
                return final_category
                
        except Exception as e:
            print(f"⚠️  Errore nel modello AI: {e}")
            # Fallback a keyword
            return self.simple_classifier.classify(text, return_confidence)
    
    def classify_batch(self, texts, return_confidence=False):
        """Classifica una lista di testi"""
        results = []
        for text in texts:
            try:
                result = self.classify(text, return_confidence)
                results.append(result)
            except Exception as e:
                print(f"Errore nel classificare '{text[:30]}...': {e}")
                fallback = ("ALTRO", 0.0) if return_confidence else "ALTRO"
                results.append(fallback)
        return results
    
    def get_model_info(self):
        """Restituisce informazioni sui modelli"""
        info = {
            "ai_model_available": self.has_ai_model,
            "simple_classifier_available": True,
            "threshold": self.use_ai_threshold,
        }
        
        if self.has_ai_model:
            try:
                ai_info = self.ai_classifier.get_model_info()
                info.update(ai_info)
            except:
                pass
        
        return info
    
    def cleanup(self):
        """Pulisce la memoria"""
        if self.has_ai_model and self.ai_classifier:
            self.ai_classifier.cleanup()

def test_hybrid_classifier():
    """Testa il classificatore ibrido"""
    print("🚀 TEST CLASSIFICATORE IBRIDO")
    print("=" * 50)
    
    classifier = HybridAIClassifier()
    
    test_cases = [
        "OpenAI GPT-4 per generazione automatica di contenuti creativi",
        "Algoritmi di computer vision per riconoscimento automatico di veicoli",
        "Robot collaborativo (cobot) per assemblaggio industriale automatizzato",
        "Tesla Model Y con sistema di guida autonoma Full Self-Driving",
        "Machine learning per analisi predittiva su big data aziendali",
        "AI per diagnosi precoce di tumori tramite imaging medico",
        "Algoritmi di deep learning e reti neurali convoluzionali",
        "Cucina italiana: ricetta autentica della carbonara romana",
        "Questo è un testo ambiguo senza parole chiave specifiche",
        "Sviluppo di software enterprise con metodologie agili",
    ]
    
    print(f"ℹ️  Modello AI disponibile: {classifier.has_ai_model}")
    print(f"ℹ️  Soglia confidenza: {classifier.use_ai_threshold}")
    print()
    
    for i, testo in enumerate(test_cases, 1):
        categoria, confidenza = classifier.classify(testo, return_confidence=True)
        
        # Simbolo basato sulla confidenza
        if confidenza >= 0.7:
            simbolo = "🟢"
        elif confidenza >= 0.4:
            simbolo = "🟡"
        else:
            simbolo = "🔴"
        
        print(f"{i:2d}. {simbolo} {categoria} ({confidenza:.1%})")
        print(f"     📝 {testo}")
        print()
    
    # Info sul sistema
    print("📊 INFORMAZIONI SISTEMA:")
    info = classifier.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    classifier.cleanup()

if __name__ == "__main__":
    test_hybrid_classifier()
