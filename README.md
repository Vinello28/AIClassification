# 🤖 AI Classification - Dockerized & Refactored

## 📋 Overview

A professional AI text classification system built with modern software engineering principles. This system classifies text into AI-related categories using a fine-tuned transformer model, packaged as a containerized microservice.

### 🏗️ Architecture

```
/
├── src/ai_classification/          # Main package
│   ├── core/                      # Core classification logic
│   │   ├── classifier.py          # Main classifier class
│   │   ├── model_utils.py         # Model management utilities
│   │   └── config.py              # Configuration settings
│   ├── api/                       # REST API components
│   │   ├── server.py              # FastAPI server
│   │   └── client.py              # API client
│   ├── data/                      # Data management
│   │   └── training_data.py       # Training datasets
│   └── utils/                     # Utility functions
├── tests/                         # Test suite
├── scripts/                       # Utility scripts
├── docker/                        # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx.conf
├── setup.py                       # Package installation
├── Makefile                       # Development commands
└── requirements.txt               # Dependencies
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Build and run with Docker Compose
cd docker
docker-compose up -d

# Or build manually
docker build -f docker/Dockerfile -t ai-classification:latest .
docker run -p 8000:8000 ai-classification:latest
```

### Option 2: Local Development

```bash
# Setup development environment
make dev-setup
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate # Windows

# Install dependencies  
make install-dev

# Start the server
make server
```

## 🔧 Development

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, will use CPU otherwise)
- Docker (for containerized deployment)

### Development Commands

```bash
make help              # Show all available commands
make install-dev       # Install in development mode
make test             # Run tests
make lint             # Check code quality
make format           # Format code
make docker-build     # Build Docker image
make server           # Start development server
```

## 📡 API Usage

### Server Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /predict` - Single text classification
- `POST /predict_batch` - Batch text classification
- `GET /docs` - Interactive API documentation

### Using the Client

```python
from src.ai_classification.api.client import AIClassificationClient

client = AIClassificationClient()

# Single prediction
result = client.predict("GPT-4 is a language model")
print(f"Category: {result['category']}, Confidence: {result['confidence']:.3f}")

# Batch predictions
texts = ["Machine learning algorithms", "Cooking recipes", "Computer vision"]
results = client.predict_batch(texts)
for r in results:
    print(f"{r['text']} → {r['category']} ({r['confidence']:.3f})")
```

### Direct HTTP Requests

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Neural networks and deep learning"}'

# Batch prediction
curl -X POST "http://localhost:8000/predict_batch" \
     -H "Content-Type: application/json" \
     -d '["AI research", "Cooking tips", "Robotics"]'
```

## 💻 Come Usare il Classificatore

### Opzione 1: Funzioni Semplici (Raccomandato)

```python
from client import classify_text, classify_texts

# Classifica un singolo testo
categoria, confidenza = classify_text("Reti neurali per computer vision")
print(f"Categoria: {categoria}, Confidenza: {confidenza:.3f}")

# Classifica più testi (più efficiente)
testi = [
    "Algoritmi di machine learning",
    "Ricette della nonna",
    "Modelli generativi GAN"
]
risultati = classify_texts(testi)
for i, (categoria, confidenza) in enumerate(risultati):
    print(f"{testi[i]} → {categoria} ({confidenza:.3f})")
```

### Opzione 2: Client Avanzato

```python
from client import AIClassificationClient

client = AIClassificationClient()

# Verifica connessione
if client.is_server_healthy():
    print("Server operativo!")
    
    # Singola predizione
    result = client.predict("Deep learning per analisi immagini")
    print(f"Categoria: {result['category']}")
    print(f"Confidenza: {result['confidence']:.3f}")
    
    # Predizioni multiple (batch)
    results = client.predict_batch([
        "Neural networks training",
        "Storia dell'arte italiana",
        "Robot industriali"
    ])
    
    for r in results:
        print(f"{r['text']} → {r['category']} ({r['confidence']:.3f})")
```

### Opzione 3: Richieste HTTP Dirette

```python
import requests

# Singola predizione
response = requests.post("http://localhost:8000/predict", 
                        json={"text": "Computer vision algorithms"})
result = response.json()
print(f"Categoria: {result['category']}")

# Predizioni multiple
response = requests.post("http://localhost:8000/predict_batch", 
                        json=["Text 1", "Text 2", "Text 3"])
results = response.json()
```

## 🔧 Integrazione in Altri Progetti

### Esempio: Elaborazione di File

```python
from client import classify_texts
import pandas as pd

# Leggi file CSV con testi da classificare
df = pd.read_csv('documenti.csv')
testi = df['contenuto'].tolist()

# Classifica tutti i testi
print(f"Classificazione di {len(testi)} documenti...")
risultati = classify_texts(testi)

# Aggiungi risultati al DataFrame
df['categoria'] = [r[0] for r in risultati]
df['confidenza'] = [r[1] for r in risultati]

# Salva risultati
df.to_csv('documenti_classificati.csv', index=False)

# Statistiche
print(df['categoria'].value_counts())
```

### Esempio: Sistema di Content Management

```python
from client import AIClassificationClient

class DocumentManager:
    def __init__(self):
        self.client = AIClassificationClient()
        self.documents = {}
    
    def add_document(self, doc_id, content):
        """Aggiunge e classifica un documento"""
        result = self.client.predict(content)
        
        self.documents[doc_id] = {
            'content': content,
            'category': result['category'],
            'confidence': result['confidence']
        }
        
        return result['category']
    
    def get_documents_by_category(self, category):
        """Ottieni tutti i documenti di una categoria"""
        return {doc_id: doc for doc_id, doc in self.documents.items() 
                if doc['category'] == category}
    
    def batch_classify(self, documents_dict):
        """Classifica più documenti in batch"""
        doc_ids = list(documents_dict.keys())
        contents = list(documents_dict.values())
        
        results = self.client.predict_batch(contents)
        
        for doc_id, result in zip(doc_ids, results):
            self.documents[doc_id] = {
                'content': result['text'],
                'category': result['category'],
                'confidence': result['confidence']
            }

# Utilizzo
dm = DocumentManager()
dm.add_document("doc1", "Algoritmi di deep learning")
dm.add_document("doc2", "Ricette tradizionali")

ai_docs = dm.get_documents_by_category("AI Generica")
print(f"Documenti AI: {len(ai_docs)}")
```

## 📊 Performance e Ottimizzazione

### Statistiche Attuali
- **Velocità**: ~3.2 predizioni/secondo
- **Memoria GPU**: Modello sempre caricato
- **Latenza**: ~300ms per singola predizione
- **Batch**: Molto più efficiente per più testi

### Suggerimenti per Ottimizzazione

1. **Usa sempre batch per più testi**:
   ```python
   # ❌ Lento
   for text in texts:
       classify_text(text)
   
   # ✅ Veloce
   classify_texts(texts)
   ```

2. **Mantieni il server sempre attivo** per evitare tempi di caricamento

3. **Per grandi volumi** (>1000 testi), dividi in batch da 50-100

## 🎯 Categorie Supportate

1. **Altro** - Contenuti non-AI
2. **AI Generica** - Machine learning, algoritmi, reti neurali
3. **AI Generativa** - GPT, DALL-E, modelli generativi
4. **Computer Vision** - Riconoscimento immagini, OCR
5. **Robotica AI** - Robot intelligenti, automazione
6. **Guida Autonoma** - Veicoli self-driving, navigazione
7. **Data Science** - Analisi dati, business intelligence
8. **AI Medica** - Diagnostica, telemedicina, bioinformatica

## 🛠️ Troubleshooting

### Il server non si avvia
```bash
# Verifica ambiente virtuale
.venv\Scripts\activate
pip install -r requirements.txt
python server.py
```

### Errore "Server non disponibile"
```python
from client import AIClassificationClient
client = AIClassificationClient()
print(client.is_server_healthy())  # Deve essere True
```

### Performance lente
- Verifica che la GPU sia utilizzata (check logs del server)
- Usa predizioni batch invece di singole
- Controlla che il modello sia in modalità eval

## 🔄 Workflow Completo

1. **Avvia server**: `python server.py`
2. **Testa connessione**: `python client.py`
3. **Integra nel tuo codice**:
   ```python
   from client import classify_text
   
   categoria, confidenza = classify_text("Il tuo testo qui")
   print(f"Risultato: {categoria} ({confidenza:.3f})")
   ```

## 📈 Monitoraggio

Il server logga automaticamente:
- Caricamento modello
- Dispositivo utilizzato (GPU/CPU)
- Errori di predizione
- Performance

Controlla i logs per eventuali problemi.

---

🎉 **Buona fortuna BBY!**

Mantieni il server attivo e potrai classificare migliaia di testi senza (troppe) perdite di performance.
