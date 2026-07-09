# Retrieval-Augmented Generation

Une application Streamlit locale pour poser des questions à des documents PDF, DOCX et TXT avec recherche hybride, reranking, citations et génération via Ollama.

Le projet est construit comme un pipeline RAG lisible plutôt que comme une démonstration opaque. Chaque étape est séparée dans un module dédié afin de pouvoir inspecter, régler et faire évoluer le système de recherche.

## Fonctionnalités

- Importer un ou plusieurs documents depuis le navigateur.
- Extraire le texte de fichiers PDF, DOCX et TXT.
- Découper les documents en chunks avec recouvrement.
- Construire un index vectoriel avec des embeddings Sentence Transformers et FAISS.
- Construire un index lexical BM25 sur les mêmes chunks.
- Fusionner les résultats vectoriels et BM25 avec Reciprocal Rank Fusion.
- Réordonner les candidats avec un cross-encoder.
- Générer une réponse avec un modèle Ollama local.
- Retourner les citations et afficher un diagnostic des chunks retrouvés puis rerankés.

## Architecture

```text
Documents
  -> loaders.py
  -> chunking.py
  -> vector_store.py + bm25.py
  -> retriever.py
  -> reranker.py
  -> prompting.py
  -> generator.py
  -> interface Streamlit
```

L'application garde la couche de recherche transparente : après chaque réponse, l'interface Streamlit peut afficher les chunks utilisés, leurs sources, leurs scores de fusion et leurs scores de reranking.

## Stack

- Python
- Streamlit
- Sentence Transformers
- FAISS
- BM25
- Cross-encoder reranking
- Ollama
- pypdf
- python-docx

## Structure du dépôt

```text
.
├── app.py                 # Interface Streamlit
├── rag/
│   ├── bm25.py            # Recherche lexicale
│   ├── chunking.py        # Découpage des documents
│   ├── config.py          # Paramètres d'exécution
│   ├── embeddings.py      # Chargement embeddings et reranker
│   ├── generator.py       # Client de génération Ollama
│   ├── loaders.py         # Parsing PDF/DOCX/TXT
│   ├── memory.py          # Formatage court de l'historique
│   ├── pipeline.py        # Pipeline RAG complet
│   ├── prompting.py       # Prompt et contexte cité
│   ├── reranker.py        # Reranking cross-encoder
│   ├── retriever.py       # Recherche hybride et fusion
│   ├── schemas.py         # Structures de données partagées
│   └── vector_store.py    # Recherche vectorielle FAISS
├── utils/
└── requirements.txt
```

## Installation

Installer les dépendances Python :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Installer et lancer Ollama, puis récupérer le modèle par défaut :

```bash
ollama pull phi3
ollama serve
```

Lancer l'application :

```bash
streamlit run app.py
```

Ouvrir l'URL locale Streamlit, importer des documents, puis poser des questions sur leur contenu.

## Configuration

Les paramètres par défaut sont dans `rag/config.py` :

- modèle d'embedding : `sentence-transformers/all-MiniLM-L6-v2`
- modèle de reranking : `cross-encoder/ms-marco-MiniLM-L-6-v2`
- modèle Ollama : `phi3`
- taille des chunks : `900`
- recouvrement des chunks : `150`
- valeurs top-k pour la recherche vectorielle, BM25, la fusion et le reranking

La barre latérale Streamlit expose les principaux paramètres de recherche au moment de l'exécution.

## Notes

Ce projet est un prototype RAG local-first. Il est utile pour comprendre et tester le pipeline de recherche, mais il n'inclut pas encore les préoccupations de production comme l'authentification, les index persistants multi-utilisateurs, l'ingestion en arrière-plan ou l'observabilité.

## Prochaines améliorations

- Ajouter des documents d'exemple et des captures d'écran.
- Ajouter des tests automatisés pour le découpage, la recherche et le formatage des citations.
- Persister les index entre les sessions.
- Ajouter des exemples d'évaluation de la qualité de recherche.
- Ajouter le streaming des réponses Ollama.
