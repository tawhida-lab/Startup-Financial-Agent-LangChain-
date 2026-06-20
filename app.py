import streamlit as pd
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_hf import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configuration de la page
st.set_page_config(page_title="Assistant Juridique - Startup Act", layout="wide")

# Sidebar pour le téléversement
with st.sidebar:
    st.header("📁 Document Source")
    uploaded_file = st.file_uploader("Téléversez le guide du Startup Act (PDF)", type="pdf")

st.title("🤖 Assistant Juridique Intelligent - Startup Act Tunisie")
st.write("Téléversez un document officiel (PDF) et posez vos questions à l'Agent IA.")

# Récupération de la clé API depuis les Secrets Streamlit
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🔑 Clé API Groq manquante dans les configurations de Streamlit Cloud.")
elif uploaded_file is not None:
    # Sauvegarde temporaire du fichier propre pour le lecteur PDF
    temp_file = "temp_uploaded_file.pdf"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.read()) # Version simplifiée et fonctionnelle

    try:
        with st.spinner("⚡ Analyse et indexation du document en cours..."):
            # 1. Chargement et découpage du PDF
            loader = PyPDFLoader(temp_file)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # 2. Création des Embeddings et de la base FAISS
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(splits, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            # 3. Configuration de l'Agent LLM via Groq
            llm = ChatGroq(temperature=0, model_name="llama3-3b-8192", groq_api_key=GROQ_API_KEY)

        st.success("✅ Document indexé avec succès ! Posez votre question ci-dessous.")

        # 4. Zone de Chat pour l'utilisateur
        user_question = st.text_input("✍️ Votre question sur le document :", placeholder="Ex: Quels sont les critères d'âge pour le label ?")

        if user_question:
            with st.spinner("🤖 L'Agent IA réfléchit..."):
                # Structure du Prompt pour éviter les hallucinations
                system_prompt = (
                    "Tu es un assistant juridique expert en droit des affaires et du Startup Act en Tunisie.\n"
                    "Réponds à la question en utilisant uniquement le contexte fourni ci-dessous. "
                    "Si tu ne connais pas la réponse ou si elle n'est pas dans le document, dis gentiment que "
                    "l'information ne se trouve pas dans le document source. Ne réinvente rien.\n\n"
                    "Contexte :\n{context}"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                # Création de la chaîne de RAG
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                # Exécution
                response = rag_chain.invoke({"input": user_question})
                
                # Affichage du résultat
                st.subheader("💡 Réponse de l'Agent :")
                st.write(response["answer"])

    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement : {e}")
    finally:
        # Nettoyage du fichier temporaire
        if os.path.exists(temp_file):
            os.remove(temp_file)
else:
    st.info("💡 Veuillez téléverser un fichier PDF dans la barre latérale pour activer l'agent.")
