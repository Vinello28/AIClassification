from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Classification Server", version="1.0.0")

# Modello globale
model = None
tokenizer = None
device = None

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    category: str

# Mapping delle categorie
CATEGORIES = {
    0: "Altro",
    1: "AI Generica", 
    2: "AI Generativa",
    3: "Computer Vision",
    4: "Robotica AI",
    5: "Guida Autonoma",
    6: "Data Science",
    7: "AI Medica"
}

@app.on_event("startup")
async def load_model():
    """Carica il modello all'avvio del server"""
    global model, tokenizer, device
    
    try:
        logger.info("Caricamento modello in corso...")
        
        # Verifica disponibilità GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Dispositivo utilizzato: {device}")
          # Carica il modello e tokenizer dai percorsi esistenti
        model_path = "./models/ai_classifier_model"
        tokenizer_path = "./models/ai_classifier_tokenizer"
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Sposta il modello sulla GPU
        model = model.to(device)
        model.eval()  # Modalità valutazione
        
        logger.info("Modello caricato con successo!")
        
    except Exception as e:
        logger.error(f"Errore nel caricamento del modello: {e}")
        raise

@app.get("/")
async def root():
    """Endpoint di base per verificare che il server funzioni"""
    return {"message": "AI Classification Server is running"}

@app.get("/health")
async def health_check():
    """Verifica lo stato del modello"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Modello non caricato")
    return {"status": "healthy", "device": str(device)}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Endpoint per la predizione"""
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Modello non caricato")
    
    try:
        # Tokenizza l'input
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        
        # Fai la predizione
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Ottieni la classe predetta e la confidenza
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = torch.max(predictions).item()
        
        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            category=CATEGORIES[predicted_class]
        )
        
    except Exception as e:
        logger.error(f"Errore nella predizione: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nella predizione: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(texts: list[str]):
    """Endpoint per predizioni multiple"""
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Modello non caricato")
    
    try:
        # Tokenizza tutti i testi
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        
        # Fai le predizioni
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Ottieni le classi predette e le confidenze
            predicted_classes = torch.argmax(predictions, dim=-1).tolist()
            confidences = torch.max(predictions, dim=-1).values.tolist()
        
        results = []
        for i, (pred_class, confidence) in enumerate(zip(predicted_classes, confidences)):
            results.append({
                "text": texts[i],
                "prediction": pred_class,
                "confidence": confidence,
                "category": CATEGORIES[pred_class]
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Errore nella predizione batch: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nella predizione: {str(e)}")

if __name__ == "__main__":
    # Avvia il server
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disabilita il reload in produzione
        workers=1      # Un solo worker per mantenere il modello in memoria
    )