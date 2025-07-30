"""
Configurazioni per il classificatore AI
"""

# Categorie di classificazione
CATEGORIES = {
    0: "ALTRO",
    1: "AI generica", 
    2: "AI generativa",
    3: "Computer Vision",
    4: "Robotica AI",
    5: "Guida Autonoma", 
    6: "Data Science",
    7: "AI Medica"
}

# Configurazioni del modello
MODEL_CONFIG = {
    "base_model": "distilbert-base-multilingual-cased",  # Modello più semplice
    "max_length": 512,
    "batch_size": 8,  # Batch size maggiore
    "num_labels": len(CATEGORIES),    "learning_rate": 1e-5,  # Learning rate più alto
    "num_epochs": 5,  # Meno epoche per evitare overfitting
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 2
}

# Configurazioni hardware
DEVICE_CONFIG = {
    "device": "cuda",
    "mixed_precision": False,  # Disabilitato per problemi di compatibilità
    "max_memory_gb": 5  # Lascia 1GB di margine sui 6GB totali
}

# Path dei modelli
MODEL_PATHS = {
    "model_dir": "./models",
    "trained_model": "./models/ai_classifier_model",
    "tokenizer": "./models/ai_classifier_tokenizer"
}

# Configurazioni di training
TRAINING_CONFIG = {
    "shuffle_data": True,  # Abilita shuffle manuale dei dati
    "shuffle_seed": 42,    # Seed per riproducibilità del shuffle
    "drop_last_batch": False,  # Non elimina l'ultimo batch se incompleto
    "pin_memory": False,   # Disabilita pin_memory per ridurre uso RAM
    "dataloader_num_workers": 0  # Numero di worker per dataloader (0 = main thread)
}
