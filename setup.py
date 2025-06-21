"""
Script di setup per il classificatore AI
"""
import subprocess
import sys
import os
import platform

def check_python_version():
    """Controlla la versione di Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Errore: Richiesto Python 3.8 o superiore")
        print(f"   Versione attuale: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def check_cuda():
    """Controlla se CUDA è disponibile"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA GPU rilevata")
            # Estrae info GPU
            lines = result.stdout.split('\n')
            for line in lines:
                if "RTX" in line or "GTX" in line or "Tesla" in line or "Quadro" in line:
                    gpu_info = line.strip()
                    print(f"   GPU: {gpu_info}")
                    break
            return True
        else:
            print("⚠️  NVIDIA GPU non rilevata - verrà utilizzata CPU")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️  CUDA/NVIDIA drivers non installati - verrà utilizzata CPU")
        return False

def install_requirements():
    """Installa le dipendenze"""
    print("\n📦 Installazione dipendenze...")
    
    try:
        # Aggiorna pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Installa PyTorch con supporto CUDA se disponibile
        if check_cuda():
            # PyTorch con CUDA per Windows
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
        else:
            # PyTorch CPU-only
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio"
            ])
        
        # Installa altre dipendenze
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ Dipendenze installate con successo!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore nell'installazione delle dipendenze: {e}")
        return False

def create_directories():
    """Crea le directory necessarie"""
    directories = ["./models", "./logs", "./data"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Directory creata: {directory}")

def test_installation():
    """Testa l'installazione"""
    print("\n🧪 Test dell'installazione...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA disponibile: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   Device CUDA: {torch.cuda.get_device_name(0)}")
            print(f"   Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
        
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
        
        import datasets
        print(f"✅ Datasets {datasets.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Errore nell'importazione: {e}")
        return False

def initial_training():
    """Avvia il training iniziale del modello"""
    print("\n🚀 Avvio training iniziale...")
    print("   Questo potrebbe richiedere alcuni minuti...")
    
    try:
        from ai_classifier import AITextClassifier
        
        # Crea e addestra il classificatore
        classifier = AITextClassifier(auto_train=True)
        
        # Test rapido
        test_text = "GPT-4 è un modello di linguaggio generativo"
        result = classifier.classify(test_text)
        print(f"✅ Test classificazione: '{test_text}' → {result}")
        
        classifier.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Errore nel training: {e}")
        return False

def main():
    """Setup principale"""
    print("🤖 Setup Classificatore AI per RTX 3060TI")
    print("=" * 50)
    
    # 1. Controlla Python
    if not check_python_version():
        sys.exit(1)
    
    # 2. Controlla CUDA
    cuda_available = check_cuda()
    
    # 3. Crea directory
    create_directories()
    
    # 4. Installa dipendenze
    if not install_requirements():
        print("\n❌ Setup fallito durante l'installazione delle dipendenze")
        sys.exit(1)
    
    # 5. Testa installazione
    if not test_installation():
        print("\n❌ Setup fallito durante il test")
        sys.exit(1)
    
    # 6. Training iniziale
    print("\n" + "=" * 50)
    user_input = input("Vuoi avviare il training iniziale del modello? (s/n): ").lower()
    
    if user_input in ['s', 'si', 'y', 'yes']:
        if initial_training():
            print("\n✅ Setup completato con successo!")
            print("\n📖 Utilizzo:")
            print("   from ai_classifier import AITextClassifier")
            print("   classifier = AITextClassifier()")
            print("   result = classifier.classify('Il tuo testo qui')")
        else:
            print("\n⚠️  Setup completato ma il training è fallito")
            print("   Puoi avviare il training manualmente in seguito")
    else:
        print("\n✅ Setup delle dipendenze completato!")
        print("   Avvia il training quando sei pronto con:")
        print("   python ai_classifier.py")
    
    print("\n📚 Consulta README.md per maggiori informazioni")

if __name__ == "__main__":
    main()
