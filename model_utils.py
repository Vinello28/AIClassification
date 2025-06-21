"""
Utilities per la gestione dei modelli AI
"""
import os
import torch
import gc
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from config import MODEL_CONFIG, DEVICE_CONFIG, MODEL_PATHS, CATEGORIES, TRAINING_CONFIG

class ModelManager:
    """Gestisce caricamento, training e salvataggio dei modelli"""
    
    def __init__(self):
        self.device = torch.device(DEVICE_CONFIG["device"] if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
        # Ottimizzazioni per GPU con memoria limitata
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Abilita mixed precision se supportato
            if DEVICE_CONFIG["mixed_precision"]:
                torch.backends.cudnn.benchmark = True
    
    def load_or_create_model(self):
        """Carica un modello esistente o ne crea uno nuovo"""
        try:
            # Prova a caricare un modello già addestrato
            if os.path.exists(MODEL_PATHS["trained_model"]):
                print("Caricamento modello addestrato...")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_PATHS["trained_model"]
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_PATHS["tokenizer"]
                )
            else:
                print("Creazione nuovo modello...")
                self._create_new_model()
                
            self.model.to(self.device)
            return True
            
        except Exception as e:
            print(f"Errore nel caricamento del modello: {e}")
            # Fallback: crea un nuovo modello
            self._create_new_model()
            return False
    
    def _create_new_model(self):
        """Crea un nuovo modello da zero"""
        print(f"Inizializzazione modello base: {MODEL_CONFIG['base_model']}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIG["base_model"]
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_CONFIG["base_model"],
            num_labels=MODEL_CONFIG["num_labels"],
            torch_dtype=torch.float16 if DEVICE_CONFIG["mixed_precision"] else torch.float32
        )
          # Aggiungi token speciali se necessario
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def train_model(self, training_data):
        """Addestra il modello sui dati forniti con shuffle automatico"""
        print("Preparazione dati di training...")
        
        # Importa random per shuffle manuale
        import random
          # Shuffle manuale dei dati di training per maggiore randomizzazione
        if TRAINING_CONFIG["shuffle_data"]:
            training_data_shuffled = training_data.copy()
            random.seed(TRAINING_CONFIG["shuffle_seed"])
            random.shuffle(training_data_shuffled)
            print(f"Dati di training mescolati con seed {TRAINING_CONFIG['shuffle_seed']}: {len(training_data_shuffled)} esempi")
        else:
            training_data_shuffled = training_data
            print(f"Shuffle disabilitato: {len(training_data_shuffled)} esempi")
        
        # Prepara i dati
        texts = [item[0] for item in training_data_shuffled]
        labels = [item[1] for item in training_data_shuffled]
        
        # Tokenizzazione
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MODEL_CONFIG["max_length"],
            return_tensors="pt"
        )
        
        # Crea dataset
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })
          # Split train/validation con shuffle
        dataset = dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)
        
        # Applica shuffle aggiuntivo se richiesto
        if TRAINING_CONFIG["shuffle_data"]:
            # Shuffle del dataset di training
            train_indices = list(range(len(dataset["train"])))
            random.seed(TRAINING_CONFIG["shuffle_seed"])
            random.shuffle(train_indices)
            dataset["train"] = dataset["train"].select(train_indices)
            print(f"Dataset di training shuffled con seed {TRAINING_CONFIG['shuffle_seed']}")
          # Configurazione training con shuffle abilitato
        training_args = TrainingArguments(
            output_dir=MODEL_PATHS["model_dir"],
            num_train_epochs=MODEL_CONFIG["num_epochs"],
            per_device_train_batch_size=MODEL_CONFIG["batch_size"],
            per_device_eval_batch_size=MODEL_CONFIG["batch_size"],
            gradient_accumulation_steps=MODEL_CONFIG["gradient_accumulation_steps"],
            warmup_steps=MODEL_CONFIG["warmup_steps"],
            weight_decay=MODEL_CONFIG["weight_decay"],
            learning_rate=MODEL_CONFIG["learning_rate"],
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",            fp16=DEVICE_CONFIG["mixed_precision"],  # Mixed precision per risparmiare memoria
            remove_unused_columns=True,
            dataloader_drop_last=TRAINING_CONFIG["drop_last_batch"],  # Non elimina l'ultimo batch anche se incompleto
            dataloader_pin_memory=TRAINING_CONFIG["pin_memory"],  # Riduce uso memoria
            dataloader_num_workers=TRAINING_CONFIG["dataloader_num_workers"],  # Worker per dataloader
        )
          # Trainer con shuffle abilitato
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            data_collator=None,  # Usa il collator di default che supporta shuffle
        )
        
        # Training
        print("Inizio training...")
        try:
            trainer.train()
            print("Training completato!")
            
            # Salva il modello
            self.save_model()
            
        except Exception as e:
            print(f"Errore durante il training: {e}")
            # Pulizia memoria GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise e
    
    def _compute_metrics(self, eval_pred):
        """Calcola metriche di valutazione"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def save_model(self):
        """Salva il modello addestrato"""
        os.makedirs(MODEL_PATHS["model_dir"], exist_ok=True)
        
        self.model.save_pretrained(MODEL_PATHS["trained_model"])
        self.tokenizer.save_pretrained(MODEL_PATHS["tokenizer"])
        
        print(f"Modello salvato in: {MODEL_PATHS['trained_model']}")
    
    def predict(self, text):
        """Predice la categoria di un testo"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Modello non caricato. Chiamare load_or_create_model() prima.")
        
        # Tokenizzazione
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MODEL_CONFIG["max_length"]
        )
        
        # Predizione
        self.model.eval()
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = torch.max(predictions).item()
        
        return predicted_class, confidence
    
    def get_memory_usage(self):
        """Restituisce l'uso della memoria GPU"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'cached': torch.cuda.memory_reserved() / 1024**3,      # GB
                'max_memory': torch.cuda.max_memory_allocated() / 1024**3  # GB
            }
        return {"gpu": "non disponibile"}
    
    def cleanup(self):
        """Pulisce la memoria GPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
