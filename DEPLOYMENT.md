# 🤖 AI Classification - Production-Ready Deployment Guide

## 🏗️ Architecture Overview

The AI Classification system has been completely refactored following software engineering best practices:

### Directory Structure
```
AIClassification/
├── src/ai_classification/          # Main application package
│   ├── core/                      # Core business logic
│   │   ├── classifier.py          # Main classifier implementation
│   │   ├── model_utils.py         # Model management utilities  
│   │   └── config.py              # Configuration management
│   ├── api/                       # REST API layer
│   │   ├── server.py              # FastAPI server implementation
│   │   └── client.py              # API client library
│   ├── data/                      # Data management
│   │   └── training_data.py       # Training datasets
│   └── utils/                     # Shared utilities
├── tests/                         # Test suites
├── scripts/                       # Utility scripts  
├── docker/                        # Container configuration
│   ├── Dockerfile                 # Production container
│   ├── docker-compose.yml         # Orchestration
│   └── nginx.conf                 # Reverse proxy config
├── Makefile                       # Development automation
├── setup.py                       # Package installation
├── requirements.txt               # Python dependencies
└── Backward compatibility wrappers
```

## 🚀 Deployment Options

### 1. Docker Deployment (Recommended)

#### Quick Start
```bash
# Clone the repository
git clone https://github.com/Vinello28/AIClassification.git
cd AIClassification

# Start with Docker Compose (includes reverse proxy)
cd docker
docker-compose up -d

# Check health
curl http://localhost/health
```

#### Manual Docker
```bash
# Build the image
docker build -f docker/Dockerfile -t ai-classification:latest .

# Run the container
docker run -d \
  --name ai-classification \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  ai-classification:latest

# Check logs
docker logs ai-classification
```

### 2. Local Development

#### Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Start development server
python server.py
```

#### Using Make Commands
```bash
make help              # Show all commands
make dev-setup         # Complete development setup
make install-dev       # Install with dev dependencies
make test             # Run test suite
make lint             # Code quality checks
make docker-build     # Build Docker image
```

## 🔧 Configuration

### Environment Variables
Copy `.env.example` to `.env` and customize:

```bash
# Application
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration  
MODEL_BASE=distilbert-base-multilingual-cased
MODEL_MAX_LENGTH=512
DEVICE=cuda  # or 'cpu'

# Paths
MODELS_DIR=./models
LOGS_DIR=./logs
```

### Hardware Requirements

#### Minimum
- CPU: 2 cores
- RAM: 4GB
- Storage: 2GB

#### Recommended
- CPU: 4+ cores
- RAM: 8GB+
- GPU: NVIDIA GPU with 4GB+ VRAM (optional)
- Storage: 5GB+

## 📡 API Reference

### Base URL
- Local: `http://localhost:8000`
- Docker: `http://localhost:8000` or `http://localhost:80` (with nginx)

### Endpoints

#### Health Check
```bash
GET /health
```
Response:
```json
{
  "status": "healthy",
  "device": "cuda:0",
  "is_trained": true
}
```

#### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "text": "Neural networks and deep learning"
}
```
Response:
```json
{
  "prediction": 1,
  "confidence": 0.95,
  "category": "AI Generica"
}
```

#### Batch Prediction
```bash
POST /predict_batch
Content-Type: application/json

["Text 1", "Text 2", "Text 3"]
```

### Categories
1. **Altro** - Non-AI content
2. **AI Generica** - General AI/ML
3. **AI Generativa** - Generative AI (GPT, DALL-E)
4. **Computer Vision** - Image recognition, OCR
5. **Robotica AI** - Intelligent robotics
6. **Guida Autonoma** - Autonomous vehicles
7. **Data Science** - Data analysis, BI
8. **AI Medica** - Medical AI applications

## 🧪 Testing

### Run Tests
```bash
# All tests
make test

# With coverage
python -m pytest tests/ --cov=src/ai_classification

# Validate refactoring
python validate_refactoring.py
```

### Test Structure
```
tests/
├── test_classifier.py     # Core classifier tests
├── test_shuffle.py        # Data shuffling tests
└── debug_test.py          # Debug utilities
```

## 🔍 Monitoring & Debugging

### Logs
```bash
# Docker logs
docker logs ai-classification

# Local logs
tail -f logs/ai_classification.log
```

### Health Monitoring
```bash
# Basic health check
curl http://localhost:8000/health

# Detailed status with script
python -c "
from src.ai_classification.api.client import AIClassificationClient
client = AIClassificationClient()
print('Server healthy:', client.is_server_healthy())
"
```

## 🔒 Security Considerations

### Production Checklist
- [ ] Change default ports if needed
- [ ] Set up proper SSL/TLS (use nginx reverse proxy)
- [ ] Configure rate limiting
- [ ] Set up monitoring and alerting
- [ ] Use secrets management for sensitive configs
- [ ] Enable container security scanning

### Network Security
```bash
# Run only on localhost in production
docker run -p 127.0.0.1:8000:8000 ai-classification

# Use nginx for SSL termination
# See docker/nginx.conf for configuration
```

## 🚀 Performance Optimization

### GPU Acceleration
```bash
# Verify GPU support
docker run --gpus all ai-classification python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
"
```

### Scaling
- Use multiple container instances behind a load balancer
- Consider GPU sharing for multiple containers
- Monitor memory usage and model loading times

## 🛠️ Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
# Ensure proper Python path
export PYTHONPATH=/app/src  # In container
export PYTHONPATH=./src     # Local development
```

#### GPU not detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Run with GPU support
docker run --gpus all ai-classification
```

#### Model loading issues
```bash
# Check model directory
ls -la models/

# Retrain if needed
python -c "from src.ai_classification.core.classifier import AITextClassifier; AITextClassifier(auto_train=True).train()"
```

## 📞 Support

For issues and questions:
1. Check the logs first
2. Run `python validate_refactoring.py` to verify setup
3. Review this documentation
4. Check GitHub issues

---

🎉 **The system is now production-ready with proper containerization and software engineering practices!**