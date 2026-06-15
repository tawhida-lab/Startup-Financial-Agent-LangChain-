# 🤖 Startup Financial & Legal Agent (Tunisia Startup Act)

Un assistant juridique et financier intelligent basé sur une architecture **RAG (Retrieval-Augmented Generation)** pour interroger les documents officiels du **Startup Act en Tunisie**. 

Cette application permet de téléverser des guides ministériels ou des documents financiers, de les indexer dans une base vectorielle locale, et de poser des questions complexes en langage naturel sans aucune hallucination.

---

🔗 **Démo en ligne :** [Cliquez ici pour tester l'application](https://startup-financial-agent-langchain-pnhnzpgs6hfnow2q9rc4ft.streamlit.app/)

---

## 🚀 Fonctionnalités
* **Analyse de documents officiels (RAG) :** Extraction précise des données juridiques et financières directement depuis des fichiers PDF.
* **Zéro Hallucination :** Le modèle répond *uniquement* en se basant sur le contexte du document fourni.
* **Interface Chat Interactive :** Une interface utilisateur moderne et fluide développée avec **Streamlit**.
* **Performance Ultra-Rapide :** Orchestration via **LangChain** et inférence instantanée grâce à l'API **Groq** avec le modèle **Llama 3.3 (70B)**.
* **Gestion du contexte Tunisien :** Optimisé pour comprendre les spécificités du Startup Act (critères d'âge de 8 ans, seuils de capital, devises en Dinar Tunisien - TND).

---

## 🛠️ Architecture Technique
* **Frontend :** Streamlit (Python 3.12)
* **Framework LLM :** LangChain (LangChain-Groq & LangChain-Core)
* **Modèle d'IA :** Llama 3.3 (70B Versatile) via Groq API
* **Embeddings :** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Base Vectorielle :** FAISS (Facebook AI Similarity Search) en mémoire informatique
* **Analyse PDF :** PyPDF & RecursiveCharacterTextSplitter

---

## 📦 Installation et Utilisation en Local

### 1. Cloner le dépôt
```bash
git clone [https://github.com/tawhida-lab/Startup-Financial-Agent-LangChain-.git](https://github.com/tawhida-lab/Startup-Financial-Agent-LangChain-.git)
cd Startup-Financial-Agent-LangChain-
