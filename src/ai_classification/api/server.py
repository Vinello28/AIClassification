from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
from typing import List
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ..core.classifier import AITextClassifier
from ..core.config import CATEGORIES

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Classification Server", version="1.0.0")

# Classificatore globale
classifier = None

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    category: str

class BatchPredictionRequest(BaseModel):
    texts: List[str]

@app.on_event("startup")
async def load_model():
    """Carica il classificatore all'avvio del server"""
    global classifier
    
    try:
        logger.info("Caricamento classificatore in corso...")
        classifier = AITextClassifier(auto_train=False)
        logger.info("Classificatore caricato con successo!")
        
    except Exception as e:
        logger.error(f"Errore nel caricamento del classificatore: {e}")
        raise

@app.get("/")
async def root():
    """Endpoint di base per verificare che il server funzioni"""
    return {"message": "AI Classification Server is running"}

@app.get("/health")
async def health_check():
    """Verifica lo stato del classificatore"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classificatore non caricato")
    
    model_info = classifier.get_model_info()
    return {
        "status": "healthy", 
        "device": model_info["device"],
        "is_trained": model_info["is_trained"]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predice la categoria di un testo"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classificatore non disponibile")
    
    try:
        # Usa il classificatore
        category, confidence = classifier.classify(request.text, return_confidence=True)
        
        # Trova l'indice della categoria
        prediction = next(k for k, v in CATEGORIES.items() if v == category)
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            category=category
        )
        
    except Exception as e:
        logger.error(f"Errore nella predizione: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nella predizione: {e}")

@app.post("/predict_batch")
async def predict_batch(texts: List[str]):
    """Predice le categorie per una lista di testi"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classificatore non disponibile")
    
    try:
        # Classifica tutti i testi
        results = classifier.classify_batch(texts, return_confidence=True)
        
        # Formatta i risultati
        formatted_results = []
        for i, (category, confidence) in enumerate(results):
            prediction = next(k for k, v in CATEGORIES.items() if v == category)
            formatted_results.append({
                "text": texts[i],
                "prediction": prediction,
                "category": category,
                "confidence": confidence
            })
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Errore nella predizione batch: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nella predizione batch: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)