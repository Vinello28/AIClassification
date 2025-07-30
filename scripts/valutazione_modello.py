"""
Test di valutazione completa del modello addestrato
"""
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_classifier import AITextClassifier
import numpy as np

def valutazione_completa():
    print("🔍 VALUTAZIONE COMPLETA DEL MODELLO")
    print("=" * 60)
    
    classificatore = AITextClassifier()
    
    # Test set completo con esempi chiari per ogni categoria
    test_cases = [
        # ALTRO (0)
        ("Ricetta tradizionale della pizza margherita napoletana", 0, "ALTRO"),
        ("Risultati campionato di calcio Serie A stagione corrente", 0, "ALTRO"),
        ("Corso di chitarra classica per principianti", 0, "ALTRO"),
        ("Vacanze estive nelle isole greche", 0, "ALTRO"),
        
        # AI generica (1)
        ("Algoritmi di machine learning per classificazione", 1, "AI generica"),
        ("Reti neurali artificiali e deep learning", 1, "AI generica"),
        ("L'intelligenza artificiale nel futuro", 1, "AI generica"),
        ("Apprendimento automatico e data mining", 1, "AI generica"),
        
        # AI generativa (2)
        ("ChatGPT genera testi automaticamente", 2, "AI generativa"),
        ("DALL-E crea immagini da descrizioni testuali", 2, "AI generativa"),
        ("GPT-4 per la scrittura creativa", 2, "AI generativa"),
        ("Stable Diffusion art generation", 2, "AI generativa"),
        
        # Computer Vision (3)
        ("Riconoscimento facciale con OpenCV", 3, "Computer Vision"),
        ("Object detection con YOLO algorithm", 3, "Computer Vision"),
        ("Classificazione automatica di immagini", 3, "Computer Vision"),
        ("OCR per estrazione testo da documenti", 3, "Computer Vision"),
        
        # Robotica AI (4)
        ("Braccio robotico industriale automatizzato", 4, "Robotica AI"),
        ("Robot domestico per pulizie autonome", 4, "Robotica AI"),
        ("Drone intelligente per delivery", 4, "Robotica AI"),
        ("Robot collaborativo in fabbrica", 4, "Robotica AI"),
        
        # Guida Autonoma (5)
        ("Tesla Model S con Autopilot attivato", 5, "Guida Autonoma"),
        ("Waymo self-driving car technology", 5, "Guida Autonoma"),
        ("Sensori LiDAR per veicoli autonomi", 5, "Guida Autonoma"),
        ("Auto senza pilota completamente autonoma", 5, "Guida Autonoma"),
        
        # Data Science (6)
        ("Big Data analytics con Apache Spark", 6, "Data Science"),
        ("Analisi predittiva su dataset aziendali", 6, "Data Science"),
        ("Machine learning per business intelligence", 6, "Data Science"),
        ("Data visualization con Python pandas", 6, "Data Science"),
        
        # AI Medica (7)
        ("Diagnosi automatica tramite imaging medicale", 7, "AI Medica"),
        ("AI per scoperta di nuovi farmaci", 7, "AI Medica"),
        ("Analisi di raggi X con deep learning", 7, "AI Medica"),
        ("Telemedicina assistita da intelligenza artificiale", 7, "AI Medica"),
    ]
    
    # Esegui i test
    risultati = []
    corretti_per_categoria = {i: {'corretti': 0, 'totali': 0} for i in range(8)}
    
    print("📊 Risultati per categoria:")
    print("-" * 60)
    
    for testo, categoria_attesa, nome_categoria in test_cases:
        categoria_predetta_id, confidenza = classificatore.model_manager.predict(testo)
        categoria_predetta_nome = classificatore.get_categories()[categoria_predetta_id]
        
        corretto = categoria_predetta_id == categoria_attesa
        risultati.append({
            'testo': testo,
            'attesa': categoria_attesa,
            'predetta': categoria_predetta_id,
            'nome_attesa': nome_categoria,
            'nome_predetta': categoria_predetta_nome,
            'confidenza': confidenza,
            'corretto': corretto
        })
        
        # Aggiorna statistiche per categoria
        corretti_per_categoria[categoria_attesa]['totali'] += 1
        if corretto:
            corretti_per_categoria[categoria_attesa]['corretti'] += 1
        
        # Stampa risultato
        stato = "✅" if corretto else "❌"
        print(f"{stato} {nome_categoria:15} → {categoria_predetta_nome:15} ({confidenza:.1%})")
    
    # Calcola metriche globali
    total_corretti = sum(1 for r in risultati if r['corretto'])
    total_test = len(risultati)
    accuratezza_globale = total_corretti / total_test
    
    print(f"\n📈 METRICHE GLOBALI:")
    print(f"   Accuratezza complessiva: {accuratezza_globale:.1%} ({total_corretti}/{total_test})")
    
    print(f"\n📋 ACCURATEZZA PER CATEGORIA:")
    categorie = classificatore.get_categories()
    for cat_id, stats in corretti_per_categoria.items():
        if stats['totali'] > 0:
            acc = stats['corretti'] / stats['totali']
            print(f"   {categorie[cat_id]:15}: {acc:.1%} ({stats['corretti']}/{stats['totali']})")
    
    # Analisi degli errori più comuni
    print(f"\n🔍 ANALISI ERRORI:")
    errori_per_categoria = {}
    
    for r in risultati:
        if not r['corretto']:
            attesa = r['nome_attesa']
            predetta = r['nome_predetta']
            if attesa not in errori_per_categoria:
                errori_per_categoria[attesa] = {}
            if predetta not in errori_per_categoria[attesa]:
                errori_per_categoria[attesa][predetta] = 0
            errori_per_categoria[attesa][predetta] += 1
    
    for categoria_attesa, errori in errori_per_categoria.items():
        print(f"   {categoria_attesa}:")
        for categoria_sbagliata, count in errori.items():
            print(f"     → classificata come {categoria_sbagliata}: {count} volte")
    
    # Test con confidenze
    print(f"\n📊 DISTRIBUZIONE CONFIDENZE:")
    confidenze = [r['confidenza'] for r in risultati]
    confidenze_corrette = [r['confidenza'] for r in risultati if r['corretto']]
    confidenze_sbagliate = [r['confidenza'] for r in risultati if not r['corretto']]
    
    print(f"   Media confidenza generale: {np.mean(confidenze):.1%}")
    if confidenze_corrette:
        print(f"   Media confidenza predizioni corrette: {np.mean(confidenze_corrette):.1%}")
    if confidenze_sbagliate:
        print(f"   Media confidenza predizioni sbagliate: {np.mean(confidenze_sbagliate):.1%}")
    
    classificatore.cleanup()
    
    return {
        'accuratezza_globale': accuratezza_globale,
        'risultati_dettagliati': risultati,
        'accuratezza_per_categoria': corretti_per_categoria
    }

def test_esempi_reali():
    """Test con esempi di testi reali più lunghi"""
    print(f"\n🌐 TEST CON ESEMPI REALI")
    print("=" * 60)
    
    classificatore = AITextClassifier()
    
    esempi_reali = [
        ("OpenAI ha rilasciato GPT-4, un modello di linguaggio multimodale che può processare sia testo che immagini. Il modello dimostra capacità umane in vari benchmark accademici e professionali.", "AI generativa"),
        
        ("I ricercatori di Google hanno sviluppato un nuovo algoritmo di computer vision che può identificare oggetti in immagini con una precisione del 95%, superando i precedenti metodi di riconoscimento automatico.", "Computer Vision"),
        
        ("Boston Dynamics presenta il nuovo robot Atlas capace di eseguire movimenti acrobatici complessi e di adattarsi a terreni irregolari grazie a sensori avanzati e algoritmi di controllo.", "Robotica AI"),
        
        ("Tesla aggiorna il software Autopilot introducendo nuove funzionalità di guida autonoma che permettono al veicolo di navigare automaticamente in città senza intervento umano.", "Guida Autonoma"),
        
        ("Una nuova ricerca utilizza algoritmi di machine learning per analizzare enormi dataset clinici e identificare pattern nascosti che possono predire malattie rare prima della manifestazione dei sintomi.", "AI Medica"),
        
        ("Le aziende stanno investendo massicciamente in soluzioni di big data analytics per ottimizzare le loro strategie di marketing e migliorare l'esperienza del cliente attraverso insights predittivi.", "Data Science"),
        
        ("Ieri sera ho preparato una cena fantastica con spaghetti alle vongole seguendo la ricetta della nonna. Gli ospiti hanno apprezzato molto il sapore autentico del piatto mediterraneo.", "ALTRO"),
    ]
    
    for testo, categoria_attesa in esempi_reali:
        categoria_predetta, confidenza = classificatore.classify(testo, return_confidence=True)
        
        stato = "✅" if categoria_predetta == categoria_attesa else "❌"
        print(f"\n{stato} Attesa: {categoria_attesa}")
        print(f"   Predetta: {categoria_predetta} ({confidenza:.1%})")
        print(f"   Testo: {testo[:100]}{'...' if len(testo) > 100 else ''}")
    
    classificatore.cleanup()

if __name__ == "__main__":
    # Esegui valutazione completa
    risultati = valutazione_completa()
    
    # Test con esempi reali
    test_esempi_reali()
    
    print(f"\n" + "=" * 60)
    print(f"🎯 CONCLUSIONE: Accuratezza finale {risultati['accuratezza_globale']:.1%}")
    
    if risultati['accuratezza_globale'] < 0.5:
        print("❌ Modello necessita miglioramenti significativi")
        print("💡 Suggerimenti:")
        print("   - Aumentare dati di training")
        print("   - Modificare architettura del modello")
        print("   - Adjustare iperparametri")
    elif risultati['accuratezza_globale'] < 0.7:
        print("⚠️  Modello funzionale ma con margini di miglioramento")
        print("💡 Suggerimenti:")
        print("   - Aggiungere più esempi per categorie problematiche")
        print("   - Fine-tuning più lungo")
    else:
        print("✅ Modello performante!")
        print("🎉 Pronto per l'uso in produzione")
