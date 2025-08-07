# AI Classification System

A professional text classification system designed to categorize content into AI-related categories using fine-tuned transformer models. The system provides both REST API and client library interfaces for seamless integration into existing applications.

## Overview

This system leverages state-of-the-art natural language processing to automatically classify text content into specific AI domains. Built with modern software engineering practices, it offers containerized deployment, comprehensive testing, and scalable architecture suitable for production environments.

### Key Features

- **Multi-category Classification**: Supports 8 distinct AI-related categories
- **High Performance**: Optimized for both single predictions and batch processing
- **REST API**: RESTful service with comprehensive documentation
- **Client Libraries**: Python client for easy integration
- **Docker Support**: Containerized deployment with Docker Compose
- **GPU Acceleration**: CUDA support for enhanced performance
- **Comprehensive Testing**: Full test suite with coverage reporting

## Architecture

```
├── src/ai_classification/          # Main package
│   ├── core/                      # Core classification logic
│   ├── api/                       # REST API components
│   ├── data/                      # Data management
│   └── utils/                     # Utility functions
├── tests/                         # Test suite
├── docker/                        # Docker configuration
├── scripts/                       # Utility scripts
└── requirements.txt               # Dependencies
```

## Quick Start

### Docker Deployment (Recommended)

```bash
# Using Docker Compose
cd docker
docker-compose up -d

# Manual Docker build
docker build -f docker/Dockerfile -t ai-classification:latest .
docker run -p 8000:8000 ai-classification:latest
```

### Local Development

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

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Detailed health status |
| POST | `/predict` | Single text classification |
| POST | `/predict_batch` | Batch text classification |
| GET | `/docs` | Interactive API documentation |

### Client Usage

```python
from src.ai_classification.api.client import AIClassificationClient

client = AIClassificationClient()

# Single prediction
result = client.predict("Machine learning algorithms optimize performance")
print(f"Category: {result['category']}, Confidence: {result['confidence']:.3f}")

# Batch predictions
texts = ["Neural network training", "Database optimization", "Computer vision"]
results = client.predict_batch(texts)
for result in results:
    print(f"{result['text']} → {result['category']} ({result['confidence']:.3f})")
```

### HTTP API

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Deep learning neural networks"}'

# Batch prediction
curl -X POST "http://localhost:8000/predict_batch" \
     -H "Content-Type: application/json" \
     -d '["AI research", "Software engineering", "Robotics"]'
```

## Development

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, CPU fallback available)
- Docker (for containerized deployment)

### Development Commands

```bash
make help              # Show all available commands
make install-dev       # Install in development mode
make test             # Run test suite
make lint             # Code quality checks
make format           # Code formatting
make docker-build     # Build Docker image
make server           # Start development server
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test
python -m pytest tests/test_classifier.py -v
```

## Supported Categories

The system classifies text into the following categories:

1. **Other** - Non-AI related content
2. **General AI** - Machine learning, algorithms, neural networks
3. **Generative AI** - GPT, DALL-E, generative models
4. **Computer Vision** - Image recognition, OCR, visual processing
5. **AI Robotics** - Intelligent robots, automation systems
6. **Autonomous Driving** - Self-driving vehicles, navigation
7. **Data Science** - Data analysis, business intelligence
8. **Medical AI** - Healthcare diagnostics, telemedicine, bioinformatics

## Performance

- **Throughput**: ~3.2 predictions/second
- **Latency**: ~300ms per single prediction
- **Batch Processing**: Significantly improved efficiency for multiple texts
- **Memory**: GPU memory optimization for model persistence

### Optimization Guidelines

1. **Use batch processing** for multiple texts:
   ```python
   # Inefficient
   for text in texts:
       client.predict(text)
   
   # Efficient
   results = client.predict_batch(texts)
   ```

2. **Keep server running** to avoid model loading overhead
3. **Process large datasets** in batches of 50-100 texts

## Integration Examples

### CSV File Processing

```python
import pandas as pd
from src.ai_classification.api.client import AIClassificationClient

client = AIClassificationClient()

# Load data
df = pd.read_csv('documents.csv')
texts = df['content'].tolist()

# Classify
results = client.predict_batch(texts)

# Add results to dataframe
df['category'] = [r['category'] for r in results]
df['confidence'] = [r['confidence'] for r in results]

# Save results
df.to_csv('classified_documents.csv', index=False)
```

### Document Management System

```python
from src.ai_classification.api.client import AIClassificationClient

class DocumentClassifier:
    def __init__(self):
        self.client = AIClassificationClient()
    
    def classify_document(self, content):
        """Classify a single document"""
        return self.client.predict(content)
    
    def classify_documents(self, documents):
        """Classify multiple documents efficiently"""
        contents = [doc['content'] for doc in documents]
        results = self.client.predict_batch(contents)
        
        for doc, result in zip(documents, results):
            doc['category'] = result['category']
            doc['confidence'] = result['confidence']
        
        return documents
```

## Troubleshooting

### Server Issues

```bash
# Check environment
source venv/bin/activate
pip install -r requirements.txt
python server.py
```

### Connection Problems

```python
from src.ai_classification.api.client import AIClassificationClient
client = AIClassificationClient()
print(client.is_server_healthy())  # Should return True
```

### Performance Issues

- Verify GPU utilization in server logs
- Use batch processing for multiple predictions
- Ensure model is in evaluation mode
- Check available system memory

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the coding standards
4. Run tests and ensure they pass (`make test`)
5. Format your code (`make format`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.