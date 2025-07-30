"""
Esempio pratico di utilizzo del classificatore AI
Questo file mostra come integrare il classificatore in un'altra applicazione
"""

from client import AIClassificationClient, classify_text, classify_texts
import time

def esempio_uso_semplice():
    """Esempio di uso con le funzioni di utilità"""
    print("=" * 60)
    print("🚀 ESEMPIO USO SEMPLICE")
    print("=" * 60)
    
    try:
        # Classificazione singola
        testo = "Algoritmi di reinforcement learning per il controllo robotico"
        categoria, confidenza = classify_text(testo)
        print(f"Testo: {testo}")
        print(f"Categoria: {categoria}")
        print(f"Confidenza: {confidenza:.4f}")
        print()
        
        # Classificazione multipla
        testi = [
            "Modelli generativi per la sintesi di immagini",
            "Ricette della cucina italiana tradizionale",
            "Analisi predittiva dei dati di vendita",
            "Chirurgia robotica per interventi mininvasivi"
        ]
        
        risultati = classify_texts(testi)
        print("Classificazioni multiple:")
        for testo, (categoria, confidenza) in zip(testi, risultati):
            print(f"- {testo[:40]}... → {categoria} ({confidenza:.3f})")
        
    except Exception as e:
        print(f"❌ Errore: {e}")

def esempio_uso_avanzato():
    """Esempio di uso avanzato con il client"""
    print("\n" + "=" * 60)
    print("🔧 ESEMPIO USO AVANZATO")
    print("=" * 60)
    
    client = AIClassificationClient()
    
    # Verifica stato server
    if not client.is_server_healthy():
        print("❌ Server non disponibile!")
        return
    
    # Esempio di elaborazione di un file di testi
    testi_da_classificare = [
        "Neural architecture search per l'ottimizzazione automatica",
        "Cucina vegana: ricette innovative",
        "Generative adversarial networks per art synthesis",
        "Computer vision per la guida autonoma",
        "Robot collaborativi nell'industria 4.0",
        "Sentiment analysis di recensioni prodotti",
        "AI per la diagnostica medica precoce",
        "Storia dell'arte rinascimentale",
        "Machine learning per trading algoritmico",
        "Fotografia digitale: tecniche avanzate"
    ]
    
    print(f"Classificazione di {len(testi_da_classificare)} testi...")
    start_time = time.time()
    
    # Uso batch per maggiore efficienza
    risultati = client.predict_batch(testi_da_classificare)
    
    end_time = time.time()
    
    if risultati:
        print(f"✅ Completato in {end_time - start_time:.2f} secondi")
        print(f"⚡ Velocità: {len(testi_da_classificare)/(end_time - start_time):.1f} testi/sec")
        print()
        
        # Analizza i risultati
        categorie_count = {}
        for risultato in risultati:
            categoria = risultato['category']
            categorie_count[categoria] = categorie_count.get(categoria, 0) + 1
        
        print("📊 Distribuzione per categoria:")
        for categoria, count in sorted(categorie_count.items()):
            print(f"  {categoria}: {count} testi")
        print()
        
        # Mostra risultati dettagliati
        print("📝 Risultati dettagliati:")
        for i, risultato in enumerate(risultati, 1):
            print(f"{i:2d}. {risultato['text'][:45]}...")
            print(f"    → {risultato['category']} (confidenza: {risultato['confidence']:.3f})")
            print()

def esempio_integrazione_applicazione():
    """Esempio di integrazione in un'applicazione"""
    print("\n" + "=" * 60)
    print("💼 ESEMPIO INTEGRAZIONE APPLICAZIONE")
    print("=" * 60)
    
    class DocumentProcessor:
        """Esempio di classe che usa il classificatore"""
        
        def __init__(self):
            self.client = AIClassificationClient()
            self.processed_docs = []
        
        def process_document(self, content: str, doc_id: str = None):
            """Processa un documento e lo classifica"""
            if not self.client.is_server_healthy():
                raise Exception("Servizio di classificazione non disponibile")
            
            result = self.client.predict(content)
            if result:
                doc_info = {
                    'id': doc_id or f"doc_{len(self.processed_docs) + 1}",
                    'content': content,
                    'category': result['category'],
                    'confidence': result['confidence'],
                    'prediction': result['prediction']
                }
                self.processed_docs.append(doc_info)
                return doc_info
            else:
                raise Exception("Errore nella classificazione")
        
        def get_documents_by_category(self, category: str):
            """Ottieni tutti i documenti di una categoria"""
            return [doc for doc in self.processed_docs if doc['category'] == category]
        
        def get_statistics(self):
            """Ottieni statistiche sui documenti processati"""
            if not self.processed_docs:
                return {}
            
            stats = {}
            for doc in self.processed_docs:
                cat = doc['category']
                if cat not in stats:
                    stats[cat] = {'count': 0, 'avg_confidence': 0}
                stats[cat]['count'] += 1
                stats[cat]['avg_confidence'] += doc['confidence']
            
            # Calcola media confidenza
            for cat in stats:
                stats[cat]['avg_confidence'] /= stats[cat]['count']
            
            return stats
    
    # Esempio di utilizzo
    processor = DocumentProcessor()
    
    # Processa alcuni documenti
    documenti = [
        ("Sviluppo di chatbot conversazionali con transformer", "doc_ai_1"),
        ("Ricette tradizionali della nonna", "doc_food_1"),
        ("Segmentazione semantica di immagini satellitari", "doc_cv_1"),
        ("Robot per la pulizia domestica automatizzata", "doc_robot_1"),
        ("Analisi dei pattern di vendita e-commerce", "doc_data_1")
    ]
    
    print("Processamento documenti...")
    for content, doc_id in documenti:
        try:
            doc_info = processor.process_document(content, doc_id)
            print(f"✅ {doc_id}: {doc_info['category']} ({doc_info['confidence']:.3f})")
        except Exception as e:
            print(f"❌ Errore processando {doc_id}: {e}")
    
    # Mostra statistiche
    print("\n📊 Statistiche:")
    stats = processor.get_statistics()
    for categoria, info in stats.items():
        print(f"  {categoria}: {info['count']} docs (confidenza media: {info['avg_confidence']:.3f})")
    
    # Esempio di query per categoria
    print(f"\n🔍 Documenti AI Generativa:")
    ai_docs = processor.get_documents_by_category("AI Generativa")
    for doc in ai_docs:
        print(f"  - {doc['id']}: {doc['content'][:50]}...")

if __name__ == "__main__":
    print("🤖 ESEMPI DI UTILIZZO DEL CLASSIFICATORE AI")
    print("=" * 60)
    print("Assicurati che il server sia attivo con: python server.py")
    print("=" * 60)
    
    # Esegui tutti gli esempi
    esempio_uso_semplice()
    esempio_uso_avanzato()
    esempio_integrazione_applicazione()
    
    print("\n" + "=" * 60)
    print("✅ ESEMPI COMPLETATI!")
    print("=" * 60)
